import math
import torch

def kl_divergence_discrete(p, q):
    p = p.detach()
    return (p * (p.log() - q.log())).sum(-1)

def kl_divergence_continuous(mu_old, std_old, mu_new, std_new):
    kl = (std_new.log() - std_old.log()) + (std_old.pow(2) + (mu_old - mu_new).pow(2)) / (2.0 * std_new.pow(2)) - 0.5
    return kl.sum(1, keepdim=True)

def surrogate_loss(new_probabilities, old_probabilities, advantages):
    return (torch.exp(new_probabilities - old_probabilities) * advantages).mean()

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
