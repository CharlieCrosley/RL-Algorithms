from contextlib import nullcontext
import math
import torch

from models.shared.core import DeterministicPolicy

def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (
        2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)


def conjugate_gradient(hessian_vector_product, b, max_iterations=10, residual_error_tolerance=1e-10):
        """ 
        Compute the Hessian-vector product using the conjugate gradient algorithm 
        Explanation: 
        - https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/BMVA96Tut/node20.html#:~:text=Two%20vectors%2C%20u%2C%20v%2C,closer%20to%20the%20minimum%20efficiently.
        - https://en.wikipedia.org/wiki/Conjugate_gradient_method (see the section on "The resulting algorithm")
        """

        x = torch.zeros(b.size(), device=b.device) # initial solution of 0
        # residual = b - torch.dot(x, self.hessian_vector_product(p, states)), equivalent to b since x is 0 at start
        residual = b.clone() # the error of the x estimate to the correct x
        p = b.clone() # search direction (points towards local minimum x_*)
        
        rdotr = residual.dot(residual)
        if rdotr < residual_error_tolerance: # if the initial residual error is small enough, return initial estimate
            return x
        
        for _ in range(max_iterations):
            # Compute the product of Hessian and search direction
            hvp = hessian_vector_product(p)
            
            # Compute the step size
            alpha = rdotr / p.dot(hvp)
            # Update the estimate
            x += alpha * p
            # Compute the new residual
            residual -= alpha * hvp
            updated_rdotr = residual.dot(residual)
            beta = updated_rdotr / rdotr
            # Update the search direction
            p = residual + beta * p
            rdotr = updated_rdotr
            if rdotr < residual_error_tolerance: 
                # if the residual error is small enough, stop
                # This means the estimate x is very close to the optimal x
                break
        return x

def estimate_advantage_with_value_fn(states, rewards, terminal, value_fn, discount=0.99):
    values = value_fn(states)
    last_value = 0
    discounted_reward = torch.empty((rewards.size(0),1), device=states.device)
    for i in reversed(range(rewards.shape[0])):
        discounted_reward[i] = rewards[i] + discount * terminal[i] * last_value
        last_value = discounted_reward[i, 0]

    returns = discounted_reward.clone()
    advantages = discounted_reward - values
    return advantages, returns

def kl_divergence_discrete(p, q):
    p = p.detach()
    return (p * (p.log() - q.log())).sum(-1)

def kl_divergence_continuous(mu_old, std_old, mu_new, std_new):
    kl = (std_new.log() - std_old.log()) + (std_old.pow(2) + (mu_old - mu_new).pow(2)) / (2.0 * std_new.pow(2)) - 0.5
    return kl.sum(1, keepdim=True)

def surrogate_loss(new_probabilities, old_probabilities, advantages):
    return (torch.exp(new_probabilities - old_probabilities) * advantages).mean()

def compute_surrogate_loss_and_kl(policy, states, actions, advantages, old_log_probs=None, eval=False, ctx=nullcontext()):
        # calculate log probabilities of actions and calculate kl divergence (combining both saves computation)
        if isinstance(policy, DeterministicPolicy): 
            with ctx:
                dist = policy(states)
            # DKL(P∥Q) is not defined if there is some i such that Q(i)=0 but P(i)≠0
            # because of this we need to clip the probabilities by nudging them away from 0 and 1 by the smallest possible value for the dtype
            dist = torch.distributions.utils.clamp_probs(dist) 
            probs = torch.gather(dist, 1, actions.long().unsqueeze(1))
            kl = None if eval else kl_divergence_discrete(dist.detach(), dist).mean()
            log_probs = probs.log()
        else:
            with ctx:
                mu_new, log_std_new = policy(states)
            std_new = torch.exp(log_std_new)
            kl = None if eval else kl_divergence_continuous(mu_new.detach(), std_new.detach(), mu_new, std_new).mean()
            log_probs = normal_log_density(actions, mu_new, log_std_new, std_new) if old_log_probs is None else old_log_probs
        
        old_log_probs = log_probs.detach() if old_log_probs is None else old_log_probs
        return surrogate_loss(log_probs, old_log_probs, advantages), log_probs, kl

def get_action_log_prob(policy, states, actions, ctx=nullcontext()):
    if isinstance(policy, DeterministicPolicy): 
        with ctx:
            dist = policy(states)
        probs = torch.gather(dist, 1, actions.long().unsqueeze(1))
        return probs.log()
    else:
        with ctx:
            mu_new, log_std_new = policy(states)
        return normal_log_density(actions, mu_new, log_std_new, torch.exp(log_std_new))

      