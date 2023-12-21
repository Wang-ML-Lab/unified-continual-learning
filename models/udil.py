import numpy as np
import torch
from torch.optim import SGD, Adam
from torch.nn.functional import softmax

from datasets import get_dataset
from backbones import get_all_backbones, get_backbone
from models.lwf import soft_ce
from models.utils.continual_model import ContinualModel, optimizer_dict

from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, add_backbone_args, ArgumentParser
from utils.buffer_feature import Buffer, setup_buffer
from utils import valid_loss
from utils.loss import CrossDomainSupConLoss

import ipdb

# Model Specific Argument Parsing
def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Unified Domain Incremental Learning.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_backbone_args(parser)
    add_rehearsal_args(parser)

    ####################################
    # for model-specific arguments.
    ####################################
    
    # Discriminator-specific hyper-parameters
    parser.add_argument('--discriminator', type=str, required=True,
                        help='Backbone name.', choices=get_all_backbones())
    parser.add_argument('--disc-hiddim', type=int, default=800,
                        help='hidden dimension of the discriminator.')
    parser.add_argument('--disc-k', type=int, default=1,
                        help='num of iterations of discriminator update.')
    parser.add_argument('--disc-lr', type=float, default=1e-3,
                        help='learning rate for the discrminator.')
    parser.add_argument('--disc-num-layers', type=int, default=3,
                        help='number of the hidden layers in the discriminator')
    parser.add_argument('--train-disc-inadv', action='store_true',
                        help='train the discriminator for one epoch before one task.')
    
    # Adaptive-loss-weights-specific hyper-parameters
    parser.add_argument('--task-weight-k', type=int, default=1,
                        help='num of iterations of task weight update.')
    parser.add_argument('--task-weight-lr', type=float, default=1e-3,
                        help='learning rate of the task weights.')
    parser.add_argument('--loop-k', type=int, default=1,
                        help='number of loops of (update_disc, update_task_weight).')
    parser.add_argument('--encoder-lambda', type=float, default=1.,
                        help='the weight of the encoder alignment loss.')
    parser.add_argument('--encoder-mu', type=float, default=1.,
                        help='the weight of the encoder past-embedding-stable loss.')
    parser.add_argument('--supcon-lambda', type=float, default=1.,
                        help='the weight of the supervised contrastive loss.')
    
    # Loss forms and VC-dimention hyper-parameters
    parser.add_argument('--C', type=float, default=1.,
                        help='the weight of the generalization error.')
    parser.add_argument('--kd-loss-form', type=str, default='ce',
                        help='The loss form of the alpha * kd_loss', choices=('ce', 'l2', '0-1'))
    parser.add_argument('--kd-threshold', type=float, default=10., 
                        help='The threshold for the LwF loss.')
    parser.add_argument('--loss-form', type=str, default='avg',
                        help='The loss form of training, sum=(crm + crm on mem);  avg: 1/t * sum.', 
                        choices=('avg', 'sum'))
    parser.add_argument('--supcon-normed', action='store_true',
                        help='whether normalize the embedding to unit sphere when doing the supervised contrastive loss')
    parser.add_argument('--supcon-temperature', type=float, default=1.,
                        help='whether normalize the embedding to unit sphere when doing the supervised contrastive loss')
    parser.add_argument('--supcon-sim', type=str, default='dotprod',
                        help='The loss form of SupCon, ', 
                        choices=('dotprod', 'l2'))
    parser.add_argument('--supcon-first-domain', action='store_true',
                        help='whether apply the supervised contrastive loss to the first-domain training.')
    parser.add_argument('--supcon-cross-domain', action='store_true',
                        help='whether apply the cross-domain SupCon loss.')
    return parser


class UDIL(ContinualModel):
    """
    Adaptive Bound Method for DIL&TIL, 
    with Rehearsal (Memory Bank) and Supervised Contrastive Loss,
    Degenerate Version.
    """
    NAME = 'multialignsupcon'

    def __init__(self, backbone, loss, args, transform):
        # optimizer self.opt created here, only for self.net object. 
        super(UDIL, self).__init__(backbone, loss, args, transform)

        self.soft = torch.nn.Softmax(dim=1)
        self.current_task = 0
        self.cpt = get_dataset(args).N_CLASSES_PER_TASK

        dataset = get_dataset(args)
        self.n_domains = dataset.N_TASKS
        # the buffer is registered using the maximum length of the dataset
        # TODO: might cause too much memory in the GPU.
        self.register_buffer("logits", torch.randn(dataset.MAX_N_SAMPLES_PER_TASK, self.cpt))

        buffer_batch_size = args.buffer_batch_size
        if args.buffer_batch_size == 0:
            buffer_batch_size = args.buffer_size

        # set up the memory buffer 
        self.memory = Buffer(
            buffer_size=args.buffer_size, 
            device=self.device, 
            input_size=dataset.INDIM, 
            num_classes=dataset.N_CLASSES_PER_TASK,
            batch_size=buffer_batch_size,
            domain_buffers=None
        ).to(self.device)

        self.disc = None
        self.disc_k = args.disc_k
        self.setup_discriminators(args, dataset)

        self.task_weight_k = args.task_weight_k
        self.loop_k = args.loop_k
        self.encoder_lambda = args.encoder_lambda
        self.encoder_mu = args.encoder_mu
        self.C = args.C

        self.past_errs = []

        # SupCon Loss
        self.supcon = CrossDomainSupConLoss(
            temperature=args.supcon_temperature, 
            base_temperature=args.supcon_temperature,
            loss_form=args.supcon_sim
        )
        self.supcon_lambda = args.supcon_lambda
        self.supcon_first = args.supcon_first_domain
        self.supcon_cross_domain = args.supcon_cross_domain

        self.lwf_threshold = args.kd_threshold

        # put self to the correct device.
        self.to(self.device)

    def begin_task(self, cur_train_loader, next_train_loader):
        # update the task number
        self.current_task += 1

        # register the number of examples in self
        self.Nt = len(cur_train_loader.dataset)

        # set up the adaptive learning rate. 
        if self.current_task > 1:
            self.setup_logits(cur_train_loader, next_train_loader)
            self.setup_task_weights()
            # self.setup_discriminators() # reset the parameters
            
            # recompute the logits, and store the embeddings for the past tasks.
            self.setup_memory()

            # iter(memory) for discriminator
            self.memory = iter(self.memory)

            # TODO: add discriminator training for one epoch before training a new task.
            if self.args.train_disc_inadv:
                self.train_discs(cur_train_loader=cur_train_loader)

            # TODO: this feature needs to be added to LwfAda
            self.reset_opt()
    
    def train_discs(self, cur_train_loader):
        """
        Attempts to stablize the encoder training.
        """
        for _, (cur_x, cur_label, cur_idx) in enumerate(cur_train_loader):
            cur_x, cur_label = cur_x.to(self.device), cur_label.to(self.device)
            cur_data = cur_x, cur_label, cur_idx

            past_data = self.get_past_data()

            err_rates, loss_disc = self.update_disc(cur_data=cur_data, past_data=past_data)
            loss_task_logits = self.update_task_weights(cur_data=cur_data, past_data=past_data, hdivs=2*(1-2*err_rates))
            # loss_task_logits = self.update_task_weights(cur_data=cur_data, past_data=past_data, hdiv=2*(1-2*loss_err_rate))
        
    def end_task(self, cur_train_loader, next_train_loader):
        setup_buffer(self, cur_train_loader, next_train_loader)
        self.setup_past_err(cur_train_loader)
        # print(self.past_errs)s

    def setup_task_weights(self):
        """Set up the task weights and the optimizer of the task weights."""
        n_prev = self.current_task - 1

        # alpha_logits
        # beta_logits 
        # gammas_logits 
        self.task_logits = torch.zeros((3, n_prev), requires_grad=True, device=self.device)
        ## at first hold the belief that (alpha, beta, gamma) = (0.47,  0.06, 0.47)
        # self.task_logits = torch.tensor(np.vstack([
        #     np.ones((1, n_prev)), 
        #     np.full((1, n_prev), fill_value=-1),
        #     np.ones((1, n_prev)) 
        # ]), dtype=torch.float, requires_grad=True, device=self.device)
        
        self.task_weight_opt = Adam([self.task_logits], lr=self.args.task_weight_lr)

    def setup_discriminators(self, args=None, dataset=None):
        """
        Here we simultaneuously align multiple domains altogether. 
        There are two advantages of this method: 
        1. guaranteeing the alignment
        2. more stable training: no need for resetting the discriminator.
        """
        if self.disc is None:
            assert args is not None and dataset is not None
            
            # get a pseudo-shape of the embedding
            self.net.eval()
            pseudo_x = torch.zeros((2, *dataset.INDIM)).to(next(self.net.parameters()).device) # in case there is BN.
            reps = self.net(pseudo_x, returnt='features')
            self.net.train()

            # create discriminator
            self.disc = get_backbone(
                backbone_name=args.discriminator, 
                indim=reps.shape[1:], 
                hiddim=args.disc_hiddim, 
                outdim=self.n_domains,
                args=args
            ).to(self.device)

        # simply reset the parameters.
        self.disc.reset_parameters()
        # optimizer: adding weight decay to avoid the exploding gradient.
        self.disc_opt = Adam(self.disc.parameters(), lr=self.args.disc_lr)

    def setup_logits(self, cur_train_loader, next_train_loader):
        """
        Need to be reformulated: apparently the alignment of the domain is not enough to include the LwF loss.
        """
        self.net.eval()
        with torch.no_grad():
            for _, (x, _, idx) in enumerate(cur_train_loader):
                # set the logits produced by the previous model
                self.logits[idx] = self.net(x.to(self.device), returnt='logits')
        self.net.train()

    def setup_past_err(self, cur_train_loader):
        """
        Update the error rates, containing two parts:
            1. Evaluate e_{D_t}(H_t) using the current-domain data
            2. Re-evaluate e_{D_i}(H_t) using the memory.
        """
        # TODO: we can add EMA to make it smoother.
        self.past_errs = []
        
        self.net.eval()
        with torch.no_grad():
            # memory of the past domains.
            for buf in self.memory.domain_buffers[:-1]:
                x, _, _, label = buf.collect_data()
                logits = self.net(x)
                error_rate = (torch.argmax(logits, 1) != label).sum().item() / x.shape[0]
                self.past_errs.append(error_rate)

            # current domain estimation is more accurate using the whole current domain data.
            error, total = 0, 0
            for _, (x, label, _) in enumerate(cur_train_loader):
                x, label = x.to(self.device), label.to(self.device)
                logits = self.net(x)
                pred = torch.argmax(logits, 1)
                error += (pred != label).sum().item()
                total += pred.shape[0]
            self.past_errs.append(error/total)
        self.net.train()

    def setup_memory(self):
        """
        In LwFAda, we need to record the embeddings (features)
        so that the update of the encoder will not influence too much
        the performance on the past tasks.
        """
        self.net.eval()
        with torch.no_grad():
            for buf in self.memory.domain_buffers:
                indices = buf.indices
                examples, _, _, _ = buf.collect_data()

                # forward of the net.
                logits, _, feats = self.net(examples, returnt='all')
                
                # update the logits and features using the current model H_{t-1}.
                buf.preds[indices] = logits
                buf.feats[indices] = feats
        self.net.train()

    def get_past_data(self):
        # get past data, past pseudo-labels, past domain ids.
        try: 
            past_data = next(self.memory)
        except:
            self.memory = iter(self.memory)
            past_data = next(self.memory)
        return past_data

    def observe(self, cur_data, next_data):
        """
        1. Update the discriminator for 1 step.
        2. Update the task weights for 1 step. 
        3. Update the network for 1 step.
        """
        past_data = None

        if self.current_task > 1:
            past_data = self.get_past_data()
            # TODO: EMA to smooth
            # ADDED: marked out the discriminator and task_weights update to catch the bug.
            for _ in range(self.loop_k):
                err_rates, loss_disc = self.update_disc(cur_data=cur_data, past_data=past_data)
                loss_task_logits = self.update_task_weights(cur_data=cur_data, past_data=past_data, hdivs=2*(1-2*err_rates))
            # print(softmax(self.task_logits, dim=0))

        # update the model parameters.
        loss_cl, return_losses = self.update_weights(cur_data=cur_data, past_data=past_data)
        
        # for loss recording
        if return_losses is not None:
            loss_cur_erm, loss_cur_kd, loss_past, loss_encoder = return_losses 
        return loss_cl
    
    def update_disc(self, cur_data, past_data):
        """Update the discriminator for a couple steps."""
        criterion = torch.nn.CrossEntropyLoss(reduction='none')

        # Now we only use the constant weights.
        n_prev = self.task_weights.shape[1]
        # task_weights = torch.tensor(np.vstack([
        #     np.zeros((1, n_prev)), 
        #     np.ones((1, n_prev)),
        #     np.zeros((1, n_prev)) 
        # ]), dtype=torch.float, device=self.device)
        task_weights = self.task_weights

        # binary classification
        for _ in range(self.disc_k):
            cur_x, _, _ = cur_data
            cur_labels = (self.current_task-1) * torch.ones((cur_x.shape[0],), dtype=torch.long, device=self.device)
            cur_sample_weights = torch.ones((cur_x.shape[0], ), device=self.device) / cur_x.shape[0]

            # get past data, past pseudo-labels, past domain ids.
            past_x, _, _, _, past_domain_ids = past_data
            past_labels = (past_domain_ids-1).type(torch.long).to(self.device)
            past_domain_ids = past_labels

            # get past sample weights.
            beta_prime = (task_weights[1] / task_weights[1].sum()).detach().clone() # grad not passed through.

            unique_labels, past_domain_cnts = torch.unique(past_domain_ids, return_counts=True, sorted=True)
            unique_labels = unique_labels.type(torch.long).to(self.device)
            past_domain_cnts = past_domain_cnts.type(torch.float).to(self.device)
            full_cnts = torch.zeros_like(beta_prime, device=self.device)
            full_cnts[unique_labels] = past_domain_cnts.type(torch.float)
            past_samples_weights = beta_prime[past_domain_ids] / full_cnts[past_domain_ids] # beta_i/N_i
            # past_samples_weights = beta_prime[past_domain_ids] / past_x.shape[0] # ADDED: make it balanced binary classification.


            # assemble the past and current to form a single batch.
            xs, labels, sample_weights = torch.vstack([cur_x, past_x]), torch.hstack([cur_labels, past_labels]), torch.hstack([cur_sample_weights, past_samples_weights])
        
            self.disc_opt.zero_grad()
            with torch.no_grad():
                # self.net.eval() # ADDED: make sure the feature computation doesn't include the BN layer and dropout output.
                features = self.net(xs, returnt='features')
                # self.net.train()
            logits = self.disc(features)
            masked_logits = logits[:, :self.current_task]
            batch_loss = criterion(masked_logits, labels)

            # beta_sum = task_weights[1].sum().detach().item() # have the same scaling effect as the encoder.
            loss = (batch_loss * sample_weights).sum()
            
            loss.backward()
            self.disc_opt.step()

        # return the final estimated error rate for each Hdiv_i. 
        # error rate: 0-1 loss
        errs = []
        # incorrects = (torch.argmax(logits, 1) != labels)
        for i in range(self.current_task - 1):
            binary_preds = torch.where(torch.argmax(logits[:, [i, self.current_task-1]], 1) == 0, i, self.current_task-1)
            incorrects = binary_preds != labels
            past_inds, cur_inds = labels == i, labels == self.current_task-1
            past_incorrects, cur_incorrects = incorrects[past_inds].sum() / past_inds.shape[0], incorrects[cur_inds].sum() / cur_inds.shape[0]
            error_rate = ((past_incorrects+cur_incorrects)/2).item()
            err = min(error_rate, 1-error_rate) # Is this correct? 
            errs.append(err)

        # print('-----discriminator norms and error-----')
        # print(f'discriminator err: {err}')
        # print([torch.norm(x).detach().item() for x in list(self.disc.parameters())])
        
        return torch.tensor(np.array(errs)).to(self.device), loss.item()

    def update_task_weights(self, cur_data, past_data, hdivs=0):
        self.net.eval()
        for _ in range(self.task_weight_k):
            self.task_weight_opt.zero_grad() 

            cur_x, cur_y, cur_idx = cur_data
            past_x, past_y, past_preds, past_feats_stored, past_domain_ids = past_data

            # concatenation for encoders that contains BN.
            inputs = torch.cat((cur_x, past_x))
            logits, _, feats = self.net(inputs, returnt='all')
            
            # split to current and past
            cur_logits, past_logits = torch.split(logits, [cur_x.shape[0], past_x.shape[0]])
            cur_feats, past_feats = torch.split(feats, [cur_x.shape[0], past_x.shape[0]])

            loss_cur_kd = self.compute_current_kd(cur_logits, cur_idx, loss_form='0-1')

            loss_past = self.compute_past_losses(
                logits_p=past_logits, 
                logits_t=past_preds, 
                labels=past_y, 
                domain_ids=past_domain_ids,
                loss_form='0-1'
            )

            # loss_cur_kd = self.compute_current_kd(cur_data=cur_data, task_weights=task_weights, loss_form='0-1')
            # loss_past = self.compute_past_losses(past_data=past_data, task_weights=task_weights, loss_form='0-1')

            loss_tradeoff = self.compute_tradeoff(hdivs=hdivs)

            loss_generalizaiton = self.generalization_error()

            # loss = 1./self.current_task * (loss_cur_erm + loss_cur_kd + loss_past + loss_tradeoff) # shouldn't be affected by the current_task number
            loss = loss_cur_kd + loss_past + loss_tradeoff + loss_generalizaiton

            loss.backward()
            # print(self.task_logits.grad)
            self.task_weight_opt.step()
            # print(softmax(self.task_logits, 0))
        self.net.train()

        return loss.item()

    def update_weights(self, cur_data, past_data):
        # first task training
        if self.current_task <= 1:
            cur_x, cur_y, _ = cur_data
            self.opt.zero_grad()
            logits, _, feats = self.net(cur_x, returnt='all')
            loss_cl = self.loss(logits, cur_y)
                    # supervised contrastive loss
            if self.supcon_first:
                loss_supcon = self.compute_supcon_loss(
                    cur_feats=feats, 
                    cur_labels=cur_y, 
                    past_feats=None, 
                    past_labels=None
                )
                assert not torch.isnan(loss_supcon) and not torch.isinf(loss_supcon)
                loss = loss_cl + loss_supcon
            else:
                loss = loss_cl
                
            loss.backward()
            self.opt.step()
            return loss_cl.item(), None
        
        # next task training
        cur_x, cur_y, cur_idx = cur_data
        past_x, past_y, past_preds, past_feats_stored, past_domain_ids = past_data

        # concatenation for encoders that contains BN.
        inputs = torch.cat((cur_x, past_x))
        logits, _, feats = self.net(inputs, returnt='all')
        
        # split to current and past
        cur_logits, past_logits = torch.split(logits, [cur_x.shape[0], past_x.shape[0]])
        cur_feats, past_feats = torch.split(feats, [cur_x.shape[0], past_x.shape[0]])

        # e_{Dt}(h)
        loss_cur_erm = self.compute_current_erm(cur_logits, cur_y)
        assert not torch.isnan(loss_cur_erm) and not torch.isinf(loss_cur_erm)

        # e_{Dt}(h, H_{t-1})
        loss_cur_kd = self.compute_current_kd(cur_logits, cur_idx, threshold=self.lwf_threshold)
        # if torch.isnan(loss_cur_kd) or torch.isinf(loss_cur_kd):
        #     loss_cur_kd = 0
        # hack the problem for now.
        assert not torch.isnan(loss_cur_kd) and not torch.isinf(loss_cur_kd)

        # gamma * E_{Di}(h) + alpha * E_{Di}(h, H_{t-1})
        loss_past = self.compute_past_losses(
            logits_p=past_logits, 
            logits_t=past_preds, 
            labels=past_y, 
            domain_ids=past_domain_ids,
            loss_form=self.args.kd_loss_form
        )
        assert not torch.isnan(loss_past) and not torch.isinf(loss_past)

        # supervised contrastive loss
        loss_supcon = self.compute_supcon_loss(
            cur_feats=cur_feats, 
            past_feats=past_feats, 
            cur_labels=cur_y, 
            past_labels=past_y, 
            cur_domains=self.current_task * torch.ones((cur_feats.shape[0], )),
            past_domains=past_domain_ids
        )
        # print(loss_supcon)
        # if torch.isnan(loss_supcon):
        #     loss_supcon = 0
        assert not torch.isnan(loss_supcon) and not torch.isinf(loss_supcon)
        
        if self.args.loss_form == 'sum':
            loss_cl = (loss_cur_erm + loss_cur_kd + loss_past)
            # loss = loss_cl = loss_cur_erm + loss_past
            # loss = loss_cl = self.loss(logits, torch.cat([cur_y, past_y]))
        elif self.args.loss_form == 'avg':
            loss_cl = 1./self.current_task * (loss_cur_erm + loss_cur_kd + loss_past)
        
        self.opt.zero_grad()
        (loss_cl+loss_supcon).backward(retain_graph=True)
        self.opt.step()


        # encoder loss:
        # 1. encoder loss that aligns embedding distribution 
        # 2. encoder loss that retains the original distribution

        # next task training
        cur_x, cur_y, cur_idx = cur_data
        past_x, past_y, past_preds, past_feats_stored, past_domain_ids = past_data

        # concatenation for encoders that contains BN.
        inputs = torch.cat((cur_x, past_x))
        logits, _, feats = self.net(inputs, returnt='all')
        
        # split to current and past
        cur_logits, past_logits = torch.split(logits, [cur_x.shape[0], past_x.shape[0]])
        cur_feats, past_feats = torch.split(feats, [cur_x.shape[0], past_x.shape[0]])

        loss_encoder = self.compute_encoder_loss(cur_feats, past_feats, past_feats_stored, past_domain_ids=past_domain_ids)
        assert not torch.isnan(loss_encoder) and not torch.isinf(loss_encoder)

        if self.args.loss_form == 'avg':
            loss_encoder /= self.current_task

        self.opt.zero_grad()
        loss_encoder.backward()
        # torch.nn.utils.clip_grad.clip_grad_value_(self.net.parameters(), 1.0)
        self.opt.step()

        # print(loss_cur_erm.item(), loss_past.item(), loss_encoder.item())

        # if self.args.loss_form s== 'sum':
        #     loss_cl = (loss_cur_erm + 0*loss_cur_kds + loss_past)
        #     loss = loss_cl + loss_encoder 
        #     # loss = loss_cl = loss_cur_erm + loss_past
        #     # loss = loss_cl = self.loss(logits, torch.cat([cur_y, past_y]))
        # elif self.args.loss_form == 'avg':
        #     loss_cl = 1./self.current_task * (loss_cur_erm + loss_cur_kd + loss_past)
        #     loss = loss_cl + 1./self.current_task * loss_encoder
        
        # self.opt.zero_grad()
        # loss.backward()
        # self.opt.step()
                
        return_losses = loss_cur_erm.item(), loss_cur_kd.item(), loss_past.item(), loss_encoder.item()
        return loss_cl.item(), return_losses # simply record the average loss.

    def compute_current_erm(self, logits, labels):
        """E_{Dt}(h)"""
        return self.loss(logits, labels)
    
    def compute_current_kd(self, logits, idx, loss_form='ce', threshold=10.):
        """E_{Dt}(h, Ht-1)"""
        task_weights = self.task_weights
        
        if loss_form == 'ce':
            loss_kd = soft_ce(logits_p=logits, logits_t=self.logits[idx, :].to(self.device), temp=1, reduction=True)
            # added a threshold mechanism that prevents pre-alignment LwF loss.
            return loss_kd * task_weights[1].sum() * float(loss_kd.item() < threshold)
        # 0-1 loss for updating beta.
        elif loss_form == '0-1':
            loss_kd = (torch.argmax(logits,1) != torch.argmax(self.logits[idx, :],1)).type(torch.float).mean()
            return loss_kd * task_weights[1].sum()
        else:
            supported_losses = ['ce', '0-1']
            raise NotImplementedError(f"Loss form '{loss_form}' not supported; currently supported: {supported_losses}")
        

    def compute_past_losses(self, logits_p, logits_t, labels, domain_ids, loss_form='ce'):
        """
        gamma * E_{Di}(h) + alpha * E_{Di}(h, Ht-1}).
            logits_p:   predicted logits on past_data;
            logits_t:   stored logits for past_data;
            labels:     stored labels for past_data;
            domain_ids: stored domain_ids for past_data;
            loss_form:  how to compute the loss between the prediction and the labels & logits;
        """
        # return self.loss(logits_p, labels)

        task_weights = self.task_weights

        # get past data, past pseudo-labels, past domain ids.
        domain_ids = (domain_ids-1).type(torch.long)

        # gamma to weight E_{Di}(h)
        alpha, gamma = task_weights[0], task_weights[2]

        # alpha_, gamma_ = task_weights[0], task_weights[2]
        # ADDED: experimental, renormalizing alpha and gamma.
        # alpha, gamma = alpha_, gamma_
        # alpha, gamma = alpha_/(alpha_ + gamma_), gamma_/(alpha_ + gamma_)

        # alpha = torch.ones_like(alpha_, dtype=torch.float, device=self.device) / 2
        # gamma = torch.ones_like(gamma_, dtype=torch.float, device=self.device) / 2

        # alpha = torch.zeros_like(alpha_, dtype=torch.float, device=self.device)
        # gamma = torch.ones_like(gamma_, dtype=torch.float, device=self.device)
        # gamma = torch.ones_like(gamma_, dtype=torch.float, device=self.device) / (self.current_task - 1)

        unique_labels, past_domain_cnts = torch.unique(domain_ids, return_counts=True, sorted=True)
        unique_labels = unique_labels.type(torch.long).to(self.device)
        past_domain_cnts = past_domain_cnts.type(torch.float).to(self.device)
        full_cnts = torch.zeros_like(gamma, device=self.device)
        full_cnts[unique_labels] = past_domain_cnts.type(torch.float)

        alpha_weights = alpha[domain_ids] / full_cnts[domain_ids] # alpha_i/N_i
        gamma_weights = gamma[domain_ids] / full_cnts[domain_ids] # gamma_i/N_i

        # losses.
        if loss_form == 'ce':
            loss_ce = (gamma_weights * self.loss(logits_p, labels, reduction='none')).sum()
            loss_kd = (alpha_weights * soft_ce(logits_p=logits_p, logits_t=logits_t, temp=1, reduction=False)).sum()
        # dark knowledge distillation for alpha.
        # ∂C/∂zi = 1/NT^2 (zi-vi), as shown in Hinton et al.
        # so we scale the kd loss by number of dimensions logits.shape[1]
        elif loss_form == 'l2':
            loss_ce = (gamma_weights * self.loss(logits_p, labels, reduction='none')).sum()
            loss_kd = (alpha_weights * torch.nn.functional.mse_loss(logits_p, logits_t, reduction='none').sum(dim=1)).sum() / logits_p.shape[1]
        # 0-1 loss for updating gamma and alphas.
        elif loss_form == '0-1':
            loss_ce = (gamma_weights * (torch.argmax(logits_p, 1) != labels).type(torch.float)).sum()
            loss_kd = (alpha_weights * (torch.argmax(logits_p, 1) != torch.argmax(logits_t,1)).type(torch.float)).sum()
        else:
            supported_losses = ['ce', '0-1', 'l2']
            raise NotImplementedError(f"Loss form '{loss_form}' not supported; currently supported: {supported_losses}")
        return loss_ce + loss_kd

    def compute_encoder_loss(self, cur_feats, past_feats, past_feats_stored, past_domain_ids, align_part='both'):
        """min encoder w.r.t. the H-divergence"""
        # Now we only use the constant weights.
        n_prev = self.task_weights.shape[1]
        # task_weights = torch.tensor(np.vstack([
        #     np.zeros((1, n_prev)), 
        #     np.ones((1, n_prev)),
        #     np.zeros((1, n_prev)) 
        # ]), dtype=torch.float, device=self.device)
        task_weights = self.task_weights
        # the binary classifierthat only aligns the current domain data 
        # to the previous representation distribution.

        ###############################################
        # Part 1: adversarial training against the discriminator
        ###############################################
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        
        # encoder (generator) loss
        # self.disc.eval()

        # fead-forward to discriminator.
        combined_feats = torch.cat([cur_feats, past_feats]) # to avoid BN in discriminator cheating
        logits = self.disc(combined_feats)
        masked_logits = logits[:, :self.current_task]

        # when calculating the adversarial loss, we minimize the largest logits, i.e., argmax.
        labels = torch.argmax(masked_logits, dim=1)
        # encoder should be scaled by the beta sum.
        # beta_sum = task_weights[1].sum().detach().item()

        # align both embedding distributions together.
        
        if align_part == 'both':
            cur_sample_weights = torch.ones((cur_feats.shape[0], ), device=self.device) / cur_feats.shape[0]
            
            # get past sample weights.
            beta_prime = (task_weights[1] / task_weights[1].sum()).detach().clone() # grad not passed through beta.
            past_domain_ids = (past_domain_ids-1).type(torch.long)

            unique_labels, past_domain_cnts = torch.unique(past_domain_ids, return_counts=True, sorted=True)
            unique_labels = unique_labels.type(torch.long).to(self.device)
            past_domain_cnts = past_domain_cnts.type(torch.float).to(self.device)
            full_cnts = torch.zeros_like(beta_prime, device=self.device)
            full_cnts[unique_labels] = past_domain_cnts.type(torch.float)
            past_samples_weights = beta_prime[past_domain_ids] / full_cnts[past_domain_ids] # beta_i/N_i
            # past_samples_weights = beta_prime[past_domain_ids] / past_feats.shape[0] # ADDED: make it balanced binary classification.

            sample_weights = torch.cat([cur_sample_weights, past_samples_weights])
            # brute-force normalize the loss.
            not_normed_loss = (sample_weights * criterion(masked_logits, labels)).sum()
            # loss = -self.encoder_lambda * beta_sum * not_normed_loss / not_normed_loss.item()
            loss = -self.encoder_lambda * not_normed_loss
        # align the current domain's embedding distribution to the memory distribution
        elif align_part == 'cur':
            cur_logits, _ = torch.split(logits, [cur_feats.shape[0], past_feats.shape[0]])
            cur_labels = torch.ones((cur_feats.shape[0],), dtype=torch.long, device=self.device)
            cur_sample_weights = torch.ones((cur_feats.shape[0], ), device=self.device) / cur_feats.shape[0]
            loss = -self.encoder_lambda * (cur_sample_weights * criterion(cur_logits, cur_labels)).sum() # already -log(D(G(x))), which is more stable.
        
        # self.disc.train()

        ###############################################
        # Part 2: encoder results on the past domains should be stable.
        ###############################################
        criterion_feat = torch.nn.MSELoss(reduction='sum')

        loss += self.encoder_mu * criterion_feat(past_feats, past_feats_stored) / past_feats.shape[0]

        return loss

    def compute_supcon_loss(self, cur_feats, cur_labels, cur_domains=None, past_feats=None, past_labels=None, past_domains=None):
        samples_per_domain = cur_feats.shape[0] if past_feats is None else past_feats.shape[0] // self.current_task
        batch_cur_feats, batch_cur_labels = torch.split(cur_feats, samples_per_domain), torch.split(cur_labels, samples_per_domain)
        if cur_domains is not None:
            batch_cur_domains = torch.split(cur_domains, samples_per_domain)

        loss = 0
        for i in range(len(batch_cur_feats)):
            loss += self._compute_supcon_loss(
                batch_cur_feats[i], 
                batch_cur_labels[i], 
                cur_domains=None if cur_domains is None else batch_cur_domains[i],
                past_feats=past_feats,
                past_labels=past_labels,
                past_domains=past_domains
            )
        
        return loss / len(batch_cur_feats)

    def _compute_supcon_loss(self, cur_feats, cur_labels, cur_domains=None, past_feats=None, past_labels=None, past_domains=None):
        domains = None
        if past_feats is None or past_labels is None:
            feats = cur_feats
            labels = cur_labels
        else:
            feats = torch.cat([cur_feats, past_feats])
            labels = torch.cat([cur_labels, past_labels])
            if cur_domains is not None and past_domains is not None:
                domains = torch.cat([cur_domains, past_domains])

        # normalized.
        if self.args.supcon_normed:
            feats = torch.nn.functional.normalize(feats).unsqueeze(1)
        else:
            feats = feats.unsqueeze(1)

        if self.supcon_cross_domain:
            loss_supcon = self.supcon(feats, labels, domains)
        else:
            loss_supcon = self.supcon(feats, labels)

        return self.supcon_lambda * loss_supcon

    def compute_tradeoff(self, hdivs):
        task_weights = self.task_weights
        beta = task_weights[1]
        past_errs = torch.tensor(self.past_errs, requires_grad=False, device=self.device)
        alpha_beta_sum = task_weights[:2].sum(0)

        return 0.5 * (beta * hdivs).sum() + alpha_beta_sum.dot(past_errs)

    def generalization_error(self):
        task_weights = self.task_weights
        num_samples = self.memory.collect_all_sizes().to(self.device)
        # print('number of samples in each memory buffer:', num_samples)
        
        term1 = (1. + task_weights[1].sum())**2 / self.Nt
        term2 = ((task_weights[0] + task_weights[2])**2 / num_samples).sum()

        return self.C * torch.sqrt(term1 + term2)
        

    def inspect_norm(self):
        with torch.no_grad():
            print('-----GRAD NORM------')
            print([torch.norm(x.grad).detach().item() for x in list(self.net.parameters())])
            print('-----WEIGHT NORM------')
            print([torch.norm(x).detach().item() for x in list(self.net.parameters())])
            print('-----------')

    @property
    def task_weights(self):
        base = torch.max(self.task_logits, 0, keepdim=True).values.detach()
        return softmax(self.task_logits-base, 0)

    def log(self, cur_train_loader, wandb):
        """customized log for each model."""
        log_dic = {}
        if self.current_task > 1:
            log_dic = {
                **{f'alpha_{i}': a for i, a in enumerate(self.task_weights[0])}, 
                **{f'beta_{i}': b for i, b in enumerate(self.task_weights[1])}, 
                **{f'gamma_{i}': c for i, c in enumerate(self.task_weights[2])}, 
            }
        wandb.log(log_dic)

    def reset_opt(self):
        self.opt = optimizer_dict[self.args.opt](self.net.parameters(), lr=self.args.lr) # opt created. 