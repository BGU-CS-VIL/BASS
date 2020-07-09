import torch
from PIL import Image
import numpy as np
import Conn_Functions as my_connectivity
from numpy.linalg import det


def initVariables():


    global IMAGE1, IMAGE2, frame0, frame1, HEIGHT, WIDTH, dx, dy, PI, d, D, C_D, N, K, K_C, loc_scale, int_scale, opt_scale, colors, LOC, ALPHA, BETA, ETA, PI_0, H, Y, X, M, M_0, HARD_EM, LOOKUP_TABLE
    global D_T, ALPHA_T, M_T, ETA_T, M_0_T, M_ETA_T, BETA_T, ETA_M_ETA_T, inf_T, log1_T, one_T, small_T, logSmall_T, WIDTH_T, idx, ne_idx, idx_all, idx_pixels, shift, shift_index, N_index, ne4_idx, ne4_idx_split, opt_scale_T
    global PI_0_T, idx_pixels_cuda, LOOKUP_TABLE_CUD0A, c_idx, C_prior, PSI_prior, NI_prior, argmax_range, ones, zeros, TrueFlow, SIGMA_INT_FLOW, SIGMA_INT, A_prior, C_prior
    global X_C_X1, X_C_X2, X_C_Y1, X_C_Y2, SIGMA_XY_X1, SIGMA_XY_X2, SIGMA_XY_Y1, SIGMA_XY_Y2
    global D_12, D_, D_Inv, det_PSI, PI, inside_padded, FrameN, c_idx_9, c_idx_25
    global Epsilon
    global ALPHA_MS, ALPHA_MS2, LOG_ALPHA_MS2
    global psi_prior, ni_prior
    global psi_prior_sons, ni_prior_sons0
    global neig_num, potts_area
    global detInt
    global dtype
    global split_lvl
    global Padding, Padding0
    global Beta_P
    global Distances_padded, Cluster_2_pixel
    global neig_padded
    global K_C_ORIGINAL
    global idx_pixels_1, idx_pixels_2, idx_pixels_3, idx_pixels_4
    global idx_1, idx_2, idx_3, idx_4
    global Plot
    global csv_file
    global Folder
    global RegSize
    global K_C_LOW, K_C_HIGH
    global Sintel, SintelSave
    global K_C_temp
    global Beta_P
    global int_scale
    global frame0
    global DP_prior
    global save_folder
    global repeat
    global split_merges
    global covarince_estimation
    global exp_name
    global csv
    global Print
    global device
    global add_splits
    global TIF_C
    global meta_data
    covarince_estimation=True
    split_merges=True
    K_C_HIGH = 999
    K_C_LOW =  0

    Sintel = False
    Padding = torch.nn.ConstantPad2d((1, 1, 1, 1), (-1)).to(device)
    Padding0 = torch.nn.ConstantPad2d((1, 1, 1, 1), 0).to(device)


    neig_num = 4
    potts_area = 25

    Beta_P = torch.from_numpy(np.array([2.7], dtype=np.float)).to(device).float()
    C_prior = 50

    ALPHA_MS = 2675 + 25*add_splits

    ALPHA_MS2 = 0.0001

    LOG_ALPHA_MS2 = -26.2


    global TIF
    if TIF:
        import tifffile
        with tifffile.TiffFile(IMAGE1) as tif:
            h_tif,w_tif = tif.pages[0].asarray().shape
            TIF_C = len(tif.pages)
            frame0 = np.zeros((h_tif,w_tif,TIF_C))
            meta_data = tif.imagej_metadata

            for c in range(len(tif.pages)):
                frame0[:,:,c] = tif.pages[c].asarray()

    else:
        frame0 = np.array(Image.open(IMAGE1).convert('RGB'))
        TIF_C = 3

    HEIGHT, WIDTH, _ = frame0.shape
    dx = np.array([[-1 / 12, 8 / 12, 0, -8 / 12, 1 / 12]])

    """
        **Global Paramter**:
        Derivation X vector
    """
    dy = np.array([[-1 / 12, 8 / 12, 0, -8 / 12, 1 / 12]]).T

    """
        **Global Paramter**:
        Derivation Y vector
    """

    """
        **Global Paramter**:
       0 Distinct color list
    """

    PI = np.pi
    """
        **Global Paramter**:
        PI math constant
    """

    d = 1
    """
        **Global Paramter**:
        Dimension of projection
    """
    D = 2
    """
        **Global Paramter**:
        Dimension of data
    """

    C_D = 3
    """
        **Global Paramter**:
        Dimension of color space
    """
    N = HEIGHT * WIDTH
    """
        **Global Paramter**:
        Number of total points
    """

    K = 4
    """
        **Glob Paramter**:
        Number of arttficial motions
    """

    K_C = 330
    """
        **Global Paramter**:
        Number of clusters
    """
    loc_scale = 1
    """
        **Global Paramter**:
        Location scaling in k-means
    """
    int_scale = 8.7

    """
        **Global Paramter**:
        Intesity scaling in k-means
    """
    opt_scale = 1
    """
        **Global Paramter**:
        Optical flow scaling in k-means
    """

    LOC = 0
    """
        **Global Paramter**
        Prior normal gamma prameters
    """

    ALPHA = 1
    """
        **Global Paramter**
        Prior normal gamma prameters
    """
    BETA = 2

    """
        **Global Paramter**
        Prior normal gamma prameters
    """

    ETA = np.ones((D, 1)) * 0
    """
        **Global Paramter**
        Prior normal gamma prameters
    """

    PI_0 = 0.00
    """
        **Global Paramter**
        Paramters of the outlier
    """

    HARD_EM = True
    """
        **Global Paramter**
        Paramters of the Hard / Soft EM
    """

    DP_prior=0.01
    """
    int: Module level variable documented inline.
    The docstring may span multiple lines. The type may optionally be specified
    on the first line, separated by a colon.
    """

    Y = np.zeros((d, N))
    H = np.zeros((N, d, D))
    X = np.zeros((D, N))
    M = np.eye(D) * (1.0 / 100) * 0
    M_0 = np.eye(d) * 100000 * 0
    LOOKUP_TABLE = torch.from_numpy(my_connectivity.Create_LookupTable()).to(device).byte()
    LOOKUP_TABLE_CUDA = my_connectivity.Create_LookupTable()

    M_T = torch.from_numpy(M).to(device).float()
    ETA_T = torch.from_numpy(ETA).to(device).float()
    M_0_T = torch.from_numpy(M_0).to(device).float()
    ALPHA_T = torch.tensor(ALPHA).to(device).float()
    D_T = torch.tensor(D).to(device).float()
    BETA_T = torch.tensor(BETA).to(device).float()
    M_ETA_T = M_T @ ETA_T.to(device).float()
    ETA_M_ETA_T = (ETA_T.transpose(0, 1) @ M_T @ (ETA_T)).to(device).float()
    inf_T = torch.tensor(np.inf, dtype=torch.float).to(device)
    log1_T = torch.from_numpy(np.array([np.log(1)], dtype=np.float)).to(device).float()
    one_T = torch.from_numpy(np.array([1], dtype=np.float)).to(device).float()
    logSmall_T = torch.from_numpy(np.array([np.log(0.0000001)], dtype=np.float)).to(device).float()
    small_T = torch.from_numpy(np.array([0.0000001], dtype=np.float)).to(device).float()
    WIDTH_T = torch.tensor(WIDTH).to(device).int()
    opt_scale_T = torch.from_numpy(np.array([opt_scale], dtype=np.float)).to(device).float()
    PI_0_T = torch.from_numpy(np.array([PI_0], dtype=np.float)).to(device).float()

    idx_all = np.arange(0, N)

    i_idx = np.zeros((4, int(N / 4) * 9))
    j_idx = np.zeros((4, int(N / 4) * 9))


    i = np.floor_divide(idx_all, WIDTH)
    j = idx_all % WIDTH
    j += 1
    i += 1
    idx_pixels1 = np.array([])
    idx_pixels2 = np.array([])
    idx_pixels3 = np.array([])
    idx_pixels4 = np.array([])
    i1 = i[np.logical_and((i % 2 == 0), (j % 2 == 0))]
    j1 = j[np.logical_and((j % 2 == 0), (i % 2 == 0))]
    idx_pixels1 = np.append(idx_pixels1, np.array([i1 - 1, i1 - 1, i1 - 1, i1, i1, i1, i1 + 1, i1 + 1, i1 + 1]) * (
                WIDTH + 2) + np.array([j1 - 1, j1, j1 + 1, j1 - 1, j1, j1 + 1, j1 - 1, j1, j1 + 1]))
    i2 = i[np.logical_and((i % 2 == 0), (j % 2 == 1))]
    j2 = j[np.logical_and((j % 2 == 1), (i % 2 == 0))]
    idx_pixels2 = np.append(idx_pixels2, np.array([i2 - 1, i2 - 1, i2 - 1, i2, i2, i2, i2 + 1, i2 + 1, i2 + 1]) * (
                WIDTH + 2) + np.array(
        [j2 - 1, j2, j2 + 1, j2 - 1, j2, j2 + 1, j2 - 1, j2, j2 + 1]))
    i3 = i[np.logical_and((i % 2 == 1), (j % 2 == 0))]
    j3 = j[np.logical_and((j % 2 == 0), (i % 2 == 1))]
    idx_pixels3 = np.append(idx_pixels3, np.array([i3 - 1, i3 - 1, i3 - 1, i3, i3, i3, i3 + 1, i3 + 1, i3 + 1]) * (
                WIDTH + 2) + np.array(
        [j3 - 1, j3, j3 + 1, j3 - 1, j3, j3 + 1, j3 - 1, j3, j3 + 1]))
    i4 = i[np.logical_and((i % 2 == 1), (j % 2 == 1))]
    j4 = j[np.logical_and((j % 2 == 1), (i % 2 == 1))]
    idx_pixels4 = np.append(idx_pixels4, np.array([i4 - 1, i4 - 1, i4 - 1, i4, i4, i4, i4 + 1, i4 + 1, i4 + 1]) * (
                WIDTH + 2) + np.array(
        [j4 - 1, j4, j4 + 1, j4 - 1, j4, j4 + 1, j4 - 1, j4, j4 + 1]))

    idx1 = (i3 - 1) * (WIDTH) + (j3 - 1)
    idx2 = (i2 - 1) * (WIDTH) + (j2 - 1)
    idx3 = (i1 - 1) * (WIDTH) + (j1 - 1)
    idx4 = (i4 - 1) * (WIDTH) + (j4 - 1)

    i = np.floor_divide(idx_all, WIDTH)
    j = idx_all % WIDTH
    i += 1
    j += 1
    inside_padded = i * (WIDTH + 2) + j
    inside_padded = torch.from_numpy(inside_padded).to(device).long()
    idx_pixels = i_idx + j_idx

    idx_all = torch.from_numpy(idx_all).to(device)
    idx_pixels_1 = torch.from_numpy(idx_pixels3).to(device).long()
    idx_pixels_2 = torch.from_numpy(idx_pixels2).to(device).long()
    idx_pixels_3 = torch.from_numpy(idx_pixels1).to(device).long()
    idx_pixels_4 = torch.from_numpy(idx_pixels4).to(device).long()

    idx_1 = torch.from_numpy(idx1).to(device)
    idx_2 = torch.from_numpy(idx2).to(device)
    idx_3 = torch.from_numpy(idx3).to(device)
    idx_4 = torch.from_numpy(idx4).to(device)

    shift = torch.from_numpy(np.array([0, 1, 2, 3, 4, 5, 6, 7])).to(device).byte().unsqueeze(0).unsqueeze(2)
    shift_index = torch.from_numpy(np.array([0, 1, 2, 3, 5, 6, 7, 8])).to(device).long()
    ne4_idx = torch.from_numpy(np.array([1, 3, 4, 5, 7])).to(device).long()
    ne4_idx_split = torch.from_numpy(np.array([-(WIDTH), -1, 0, 1, WIDTH])).to(device).long()
    N_index = torch.arange(0, N).to(device)

    c_idx = np.zeros((N, 5))

    for i in range(0, HEIGHT):
        for j in range(0, WIDTH):
            temp_idx = (i) * (WIDTH) + j

            # up
            if (i == 0):
                c_idx[temp_idx, 0] = temp_idx
            else:
                c_idx[temp_idx, 0] = temp_idx - WIDTH

            # left
            if (j == 0):
                c_idx[temp_idx, 1] = temp_idx
            else:
                c_idx[temp_idx, 1] = temp_idx - 1

            # down
            if (i == (HEIGHT - 1)):
                c_idx[temp_idx, 2] = temp_idx
            else:
                c_idx[temp_idx, 2] = temp_idx + WIDTH

            # right
            if (j == (WIDTH - 1)):
                c_idx[temp_idx, 3] = temp_idx
            else:
                c_idx[temp_idx, 3] = temp_idx + 1

            c_idx[temp_idx, 4] = temp_idx

    c_idx = c_idx[:, 0:neig_num]
    c_idx = torch.from_numpy(c_idx).to(device).long().reshape(-1)

    c_idx_9 = np.zeros((N, 9))

    for i in range(0, HEIGHT):
        for j in range(0, WIDTH):
            temp_idx = (i) * (WIDTH) + j

            if (i == 1 and j == 0):
                b = 3
            # up
            if (i == 0):
                c_idx_9[temp_idx, 0] = temp_idx
            else:
                c_idx_9[temp_idx, 0] = temp_idx - WIDTH

            # left
            if (j == 0):
                c_idx_9[temp_idx, 1] = temp_idx
            else:
                c_idx_9[temp_idx, 1] = temp_idx - 1

            # down
            if (i == (HEIGHT - 1)):
                c_idx_9[temp_idx, 2] = temp_idx
            else:
                c_idx_9[temp_idx, 2] = temp_idx + WIDTH

            # right
            if (j == (WIDTH - 1)):
                c_idx_9[temp_idx, 3] = temp_idx
            else:
                c_idx_9[temp_idx, 3] = temp_idx + 1

            # up_left
            if (i == 0):
                c_idx_9[temp_idx, 4] = temp_idx
            else:
                if (j > 0):
                    c_idx_9[temp_idx, 4] = temp_idx - WIDTH - 1
                else:
                    c_idx_9[temp_idx, 4] = temp_idx

            # up_right
            if (i == 0):
                c_idx_9[temp_idx, 5] = temp_idx
            else:
                if (j < WIDTH - 1):
                    c_idx_9[temp_idx, 5] = temp_idx - WIDTH + 1
                else:
                    c_idx_9[temp_idx, 5] = temp_idx

            # down left
            if (i == (HEIGHT - 1)):
                c_idx_9[temp_idx, 6] = temp_idx
            else:
                if (j > 0):
                    c_idx_9[temp_idx, 6] = temp_idx + WIDTH - 1
                else:
                    c_idx_9[temp_idx, 6] = temp_idx

            # down right
            if (i == (HEIGHT - 1)):
                c_idx_9[temp_idx, 7] = temp_idx
            else:
                if (j < WIDTH - 1):
                    c_idx_9[temp_idx, 7] = temp_idx + WIDTH + 1
                else:
                    c_idx_9[temp_idx, 7] = temp_idx

            c_idx_9[temp_idx, 8] = temp_idx

            for m in range(0, 9):
                if (c_idx_9[temp_idx, m] == -1):
                    b = 5
    c_idx_9 = torch.from_numpy(c_idx_9).to(device).long().reshape(-1)

    matrix = np.arange(0, N).reshape(HEIGHT, WIDTH)
    padded_matrix = np.pad(matrix, 5, pad_with, padder=-1)
    c_idx_25 = np.zeros((N, 25))
    for i in range(5, 5 + HEIGHT):
        for j in range(5, 5 + WIDTH):
            temp_idx = (i - 5) * (WIDTH) + j - 5
            c_idx_25[temp_idx] = padded_matrix[i - 2:i + 3, j - 2:j + 3].reshape(-1)
            c_idx_25[temp_idx] = np.where(c_idx_25[temp_idx] == -1, padded_matrix[i, j], c_idx_25[temp_idx])

    c_idx_25 = torch.from_numpy(c_idx_25).to(device).long().reshape(-1)
    padded_matrix = np.pad(matrix, 7, pad_with, padder=-1)

    if (potts_area == 49):

        c_idx_25 = np.zeros((N, 49))
        for i in range(7, 7 + HEIGHT):
            for j in range(7, 7 + WIDTH):
                temp_idx = (i - 7) * (WIDTH) + j - 7
                c_idx_25[temp_idx] = padded_matrix[i - 3:i + 4, j - 3:j + 4].reshape(-1)
                c_idx_25[temp_idx] = np.where(c_idx_25[temp_idx] == -1, padded_matrix[i, j], c_idx_25[temp_idx])

        c_idx_25 = torch.from_numpy(c_idx_25).to(device).long().reshape(-1)

    A_prior = N / (K_C)
    PSI_prior = A_prior * A_prior * np.eye(2)
    det_PSI = torch.from_numpy(np.array([det(PSI_prior)])).to(device).float()
    NI_prior = C_prior * A_prior
    C_prior = torch.from_numpy(np.array([C_prior], dtype=np.float)).to(device).float()
    NI_prior = torch.from_numpy(np.array([NI_prior - 3], dtype=np.float)).to(device).float()
    PSI_prior = torch.from_numpy(PSI_prior).to(device).float().reshape(-1)
    argmax_range = torch.from_numpy(np.arange(0, N * (K_C + 1), K_C + 1)).to(device)
    ones = torch.ones(N).to(device)
    zeros = torch.zeros(N).to(device)
    X_C_X1 = torch.arange(0, N * 5 * 12, 12).to(device)
    X_C_X2 = torch.arange(5, N * 5 * 12, 12).to(device)
    X_C_Y1 = torch.arange(1, N * 5 * 12, 12).to(device)
    X_C_Y2 = torch.arange(6, N * 5 * 12, 12).to(device)

    SIGMA_XY_X1 = torch.cat((torch.arange(0, N * 5 * 8, 8), torch.arange(1, N * 5 * 8, 8))).to(device)
    SIGMA_XY_Y1 = torch.cat((torch.arange(2, N * 5 * 8, 8), torch.arange(3, N * 5 * 8, 8))).to(device)
    SIGMA_XY_X2 = torch.cat((torch.arange(4, N * 5 * 8, 8), torch.arange(5, N * 5 * 8, 8))).to(device)
    SIGMA_XY_Y2 = torch.cat((torch.arange(6, N * 5 * 8, 8), torch.arange(7, N * 5 * 8, 8))).to(device)

    a = 3
    SIGMA_INT_FLOW = torch.from_numpy(
        np.array([1.0 / int_scale, 1.0 / int_scale, 1.0 / int_scale, 1.0 / opt_scale, 1.0 / opt_scale])).unsqueeze(
        0).unsqueeze(0).float().to(device)
    SIGMA_INT = np.repeat(np.array(1.0 / int_scale),TIF_C)

    SIGMA_INT = torch.from_numpy(SIGMA_INT).unsqueeze(0).unsqueeze(0).float().to(device)
    PI = torch.from_numpy(np.array([np.pi])).float().to(device)
    D_ = TIF_C + 2 # prev - 5
    D_Inv = 4

    Epsilon = torch.zeros(N).to(device).float() + 0.000000001
    split_lvl = torch.zeros(N).to(device).float()
    Distances_padded = torch.zeros(2, ((HEIGHT + 2) * (WIDTH + 2))).float().to(device).transpose(0, 1)
    Cluster_2_pixel = torch.zeros(2, ((HEIGHT + 2) * (WIDTH + 2))).float().to(device).transpose(0, 1)
    Cluster_2_pixel[:, 0] = torch.arange(0, (HEIGHT + 2) * (WIDTH + 2)).int().to(device)
    neig_padded = torch.from_numpy(np.array([-1, 1, -(WIDTH + 2), (WIDTH + 2)])).to(device).int()

    detInt = int_scale * int_scale * int_scale
    detInt = torch.from_numpy(np.array([detInt], dtype=np.float)).to(device).float()


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector
