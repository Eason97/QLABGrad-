import copy
import torch
import re
_device = torch.device("cuda")
_is_dict = False
_trans_name_dict = None
def translate_var_name(model):
    num_pat = re.compile(r'\.([0-9]+)\.')
    trans_name = {}
    for ws in model.state_dict():
        nums = num_pat.findall(ws)
        new_name = ws
        if len(nums) > 0:
            for i in range(len(nums)):
                th = '[%s].' % nums[i]
                new_name = re.sub(r'\.' + nums[i] + '\.', th, new_name)
        trans_name[ws] = new_name
    return trans_name
def flat_grad(grad_params):
    views = []
    for ws in grad_params:
        if grad_params[ws] is not None:
            view =grad_params[ws].view(-1)
            views.append(view)
    grads = torch.cat(views, 0)
    return grads
def qlab_delta(optimizer, model, criterion, xtrain, ytrain,beta):
    global _device
    global _is_dict
    global _trans_name_dict
    if not _is_dict:
        _trans_name_dict = translate_var_name(model)

    _output = model(xtrain)
    loss = criterion(_output, ytrain)
    loss_value = loss.detach().cpu().item()
    a0= loss_value


    original_pars = copy.deepcopy(model.state_dict())
    gbeta_parameters = copy.deepcopy(original_pars)
    temp_grads = {}
    for ws in original_pars:
        expr = 'temp_grads["%s"]=copy.deepcopy(model.%s.grad)' % (ws, _trans_name_dict[ws])
        exec(expr, {'temp_grads': temp_grads, 'copy': copy, 'model': model})

    max_step = -1
    for ws in original_pars:
        if temp_grads[ws] is not None:
            scale = max(temp_grads[ws].max(), abs(temp_grads[ws].min()))
            # temp_grads[ws] = temp_grads[ws] / scale
            if scale > max_step:
                max_step = scale
    for ws in original_pars:
        if temp_grads[ws] is not None:
            temp_grads[ws] = temp_grads[ws] / max_step


    cur_gradient = flat_grad(temp_grads)
    a1 = -torch.dot(cur_gradient, cur_gradient)

    for ws in original_pars:
        if temp_grads[ws] is not None:
            # theta_minus = theta - beta * grad
            gbeta_parameters[ws] = original_pars[ws].add(-beta, temp_grads[ws])
    with torch.no_grad():
        _tp_model = copy.deepcopy(model).to(_device)
        _tp_model.load_state_dict(gbeta_parameters)
        _tp_model.eval()
        gbeta_output = _tp_model(xtrain)
        g_beta_loss = criterion(gbeta_output, ytrain)
        g_beta = g_beta_loss.item()
    s=1
    a2=(g_beta-a0-beta*a1)/(s*beta*beta)

    _e = 1e-24
    delta_opt=-a1/(2*a2+_e)
    if delta_opt<0:
        delta_opt=beta
    delta_opt = min(abs(delta_opt), 0.5)
    optimizer.param_groups[0]['lr'] = delta_opt
    return delta_opt
