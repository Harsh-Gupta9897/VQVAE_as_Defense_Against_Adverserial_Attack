import torch 

def fgsm_attack(x,epsilon,data_grad):
    pert_out = x + epsilon*data_grad.sign()
    pert_out = torch.clamp(pert_out, 0, 1)
    return pert_out

def ifgsm_attack(x,epsilon,data_grad):
    epoch = 10
    alpha = epsilon/epoch
    pert_out = x
    for i in range(epoch-1):
        pert_out = pert_out + alpha*data_grad.sign()
        pert_out = torch.clamp(pert_out, 0, 1)
        if torch.norm((pert_out-x),p=float('inf')) > epsilon: break
    return pert_out

def mifgsm_attack(x,epsilon,data_grad):
    epoch=10
    decay_factor=1.0
    pert_out = x
    alpha = epsilon/epoch
    g=0
    for i in range(epoch-1):
        g = decay_factor*g + data_grad/torch.norm(data_grad,p=1)
        pert_out = pert_out + alpha*torch.sign(g)
        pert_out = torch.clamp(pert_out, 0, 1)
        if torch.norm((pert_out-x),p=float('inf')) > epsilon: break
    return pert_out





# def pgd_attack(x, model,epsilon, data_grad, alpha=0.01, num_iter=40):
#     pert_out = x.detach() + torch.zeros_like(x).uniform_(-epsilon, epsilon)
#     pert_out = torch.clamp(pert_out, 0, 1)
#     for i in range(num_iter):
#         pert_out.requires_grad = True
#         loss = F.cross_entropy(model(pert_out), target)
#         loss.backward()
#         with torch.no_grad():
#             pert_out += alpha * pert_out.grad.sign()
#             pert_out = torch.min(torch.max(pert_out, x - epsilon), x + epsilon)
#             pert_out = torch.clamp(pert_out, 0, 1)
#         pert_out.grad.zero_()
#         if torch.norm((pert_out - x), p=float('inf')) > epsilon:
#             break
#     return pert_out

# def momentum_pgd_attack(x, model,epsilon, data_grad, alpha=0.01, mu=0.9, num_iter=40):
#     pert_out = x.detach() + torch.zeros_like(x).uniform_(-epsilon, epsilon)
#     pert_out = torch.clamp(pert_out, 0, 1)
#     delta = torch.zeros_like(x)
#     for i in range(num_iter):
#         pert_out.requires_grad = True
#         loss = F.cross_entropy(model(pert_out), target)
#         loss.backward()
#         with torch.no_grad():
#             delta = mu * delta + alpha * pert_out.grad.sign()
#             pert_out += delta
#             pert_out = torch.min(torch.max(pert_out, x - epsilon), x + epsilon)
#             pert_out = torch.clamp(pert_out, 0, 1)
#         pert_out.grad.zero_()
#         if torch.norm((pert_out - x), p=float('inf')) > epsilon:
#             break
#     return pert_out




# def deepfool_attack(x, epsilon, net, num_classes=10, max_iter=50):
#     x = x.unsqueeze(0)  # add batch dimension
#     x_adv = x.clone().detach()
#     x.requires_grad = True
#     fs = net(x)[0]  # forward pass
#     _, labels_orig = torch.max(fs, 0)  # get original label
#     num_channels = x.shape[1]
#     for _ in range(max_iter):
#         fs = net(x_adv)[0]  # forward pass on perturbed sample
#         gradient_matrix = torch.zeros(num_classes, num_channels, *x.shape[2:]).to(x.device)
#         for k in range(num_classes):
#             if k == labels_orig:
#                 continue
#             net.zero_grad()
#             fs[0, k].backward(retain_graph=True)
#             gradient_matrix[k] = x.grad.data  # compute gradient
#         y_pred = fs.argmax()
#         if y_pred == labels_orig:
#             break
#         w = torch.zeros(num_classes)
#         f = torch.zeros(num_classes)
#         for k in range(num_classes):
#             if k == labels_orig:
#                 continue
#             w[k] = gradient_matrix[k].flatten().dot((x_adv - x).flatten())
#             f[k] = fs[0, k] - fs[0, labels_orig]
#         p = torch.abs(w) / torch.norm(gradient_matrix.view(num_classes, -1), dim=1)
#         k_min = torch.argmin(p)
#         pert = (torch.abs(w[k_min]) / torch.norm(gradient_matrix[k_min])) ** 2 * (x_adv - x)
#         x_adv += pert
#         x_adv = torch.clamp(x_adv, x - epsilon, x + epsilon)
#     return x_adv.squeeze(0)