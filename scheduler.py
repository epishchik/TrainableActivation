import torch.optim.lr_scheduler as lr_scheduler


def Scheduler(optimizer, parameters):
    num = parameters['num']

    if num == 1:
        name = parameters['scheduler0']['name']
        params = parameters['scheduler0']['parameters']

        scheduler = getattr(lr_scheduler, name)
        return scheduler(optimizer, **params)
    elif num > 1:
        scheds = []
        milestones = parameters['milestones']

        for i in range(num):
            name = parameters[f'scheduler{i}']['name']
            params = parameters[f'scheduler{i}']['parameters']

            scheduler = getattr(lr_scheduler, name)
            scheds.append(scheduler(optimizer, **params))

        return lr_scheduler.SequentialLR(optimizer,
                                         schedulers=scheds,
                                         milestones=milestones)


def print_scheduler(scheduler, parameters):
    num = parameters['num']
    res_str = ''

    if num == 1:
        name = parameters['scheduler0']['name']
        state_dict = scheduler.state_dict()

        res_str += f'\n{name} (\n'
        for key in state_dict:
            res_str += f'  {key}: {state_dict[key]}\n'
        res_str += ')'
    elif num > 1:
        name = 'SequentialLR'
        state_dict = scheduler.state_dict()

        names = [parameters[f'scheduler{i}']['name'] for i in range(num)]
        res_str += f'\n{name} (\n'
        for key in state_dict:
            if key == '_schedulers':
                res_str += f'  {key}: [\n'
                for i, sched in enumerate(state_dict[key]):
                    res_str += f'    {names[i]} (\n'
                    for sched_key in sched:
                        res_str += f'      {sched_key}: {sched[sched_key]}\n'
                    res_str += '    ),\n'
                res_str += '  ]\n'
            else:
                res_str += f'  {key}: {state_dict[key]}\n'
        res_str += ')'
    return res_str
