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
        # print(ws)
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

def FindPLR(g_0,init_lr,model, criterion, xtrain, ytrain):
    global _device
    global _is_dict
    global _trans_name_dict
    if not _is_dict:
        _trans_name_dict = translate_var_name(model)

    original_parmeters = copy.deepcopy(model.state_dict())

    galpha_parameters = copy.deepcopy(original_parmeters)
    temp_grads = {}
    for ws in original_parmeters:
        expr = 'temp_grads["%s"]=copy.deepcopy(model.%s.grad)' % (ws, _trans_name_dict[ws])
        exec(expr, {'temp_grads': temp_grads, 'copy': copy, 'model': model})

    for ws in original_parmeters:
        if temp_grads[ws] is not None:
            galpha_parameters[ws] = original_parmeters[ws].add(-init_lr, temp_grads[ws])
    with torch.no_grad():
        _tp_model = copy.deepcopy(model).to(_device)
        _tp_model.load_state_dict(galpha_parameters)
        _tp_model.eval()
        gbeta_output = _tp_model(xtrain)
        g_alpha_loss = criterion(gbeta_output, ytrain)
        g_alpha =g_alpha_loss.item()
    cur_gradient = flat_grad(galpha_parameters)
    cur_gradient=torch.dot(cur_gradient,cur_gradient)
    a2=g_alpha-g_0+init_lr*cur_gradient
    # bad condition1: too big starting learning rate
    # bad condition2: too small starting learning rate
    if a2<0:
        cur_lr=2*init_lr
        return cur_lr
    if g_alpha>g_0:
        cur_lr = init_lr / 2
        return cur_lr





