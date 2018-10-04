import copy

def Recover(Data, Bound_List):
    Data = Data.cpu().data.numpy()
    # Recover the data
    for row in range (0, Data.shape[0]):
        for col in range(0, Data.shape[1]):                           
            Data[row, col] = Data[row, col] * (Bound_List[col][0] - Bound_List[col][1]) + Bound_List[col][1]
    return Data

def grid_search(lower_bound, upper_bound, step, n = 'A', grid = [], grids = []):
    if n == 'A':
        n = len(lower_bound) - 1
    if not grid:
        grid = copy.deepcopy(lower_bound)
        grids.append(copy.deepcopy(grid))
    if grid != upper_bound:
        grid[n] = grid[n] + step
        if grid[n] > upper_bound[n]:
            grid[n] = lower_bound[n]
            grid_search(lower_bound, upper_bound, step, n = n - 1, grid = grid, grids = grids)
        else:
            grids.append(copy.deepcopy(grid))
            grid_search(lower_bound, upper_bound, step, n = len(lower_bound) - 1, grid = grid, grids = grids)
    return grids

def Grid_search(Net_alpha, Net_omega, step):
    Grids = []
    for n in range(len(Net_alpha), len(Net_omega) + 1):
        init = [Net_alpha[0]] * n
        end = [Net_omega[0]] * n
        Grid = grid_search(init, end, step, grids = [])
        Grids = Grids + Grid
    return Grids

if __name__  ==  '__main__':
    lower_bound = [100, 300, 500]
    upper_bound = [200, 400, 600]
    step = 25
    networks = grid_search(lower_bound, upper_bound, step, grids = [])
    Net_alpha = [100, 100]
    Net_omega = [300, 300, 300]
    step = 25
    Networks = Grid_search(Net_alpha, Net_omega, step)