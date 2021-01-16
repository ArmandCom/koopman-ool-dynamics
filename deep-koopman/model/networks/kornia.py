import torch
# import kornia
import torch.nn.functional as F

# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

def test_kornia(x):

    x = x.reshape(-1, *x.shape[-2:])[0:1, None]
    # the source points are the region to crop corners
    points_src = torch.FloatTensor([[
        [20, 20], [50, 20], [50, 50], [20, 50],
    ]])
    # print(points_src.shape)
    for i in range(points_src.shape[1]):
        x[:, :, points_src[0,i,0].long(), points_src[0,i,1].long()] = 1

    # the destination points are the image vertexes
    h, w = 32, 32  # destination size
    points_dst = torch.FloatTensor([[[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1],]])

    # compute perspective transform
    M = kornia.get_perspective_transform(points_src, points_dst)
    M_inv = kornia.get_perspective_transform(points_dst, points_src)

    # warp the original image by the found transform
    img_warp = kornia.warp_perspective(x*255, M, dsize=(h, w))
    x_fin= kornia.warp_perspective(img_warp, M_inv, dsize=(h*2, w*2))

    print(M)

    # convert back to numpy
    image_warp = kornia.tensor_to_image(img_warp.byte()[0])

    # %matplotlib inline

    # create the plot
    fig, axs = plt.subplots(1, 3, figsize=(23, 10))
    axs = axs.ravel()

    axs[0].axis('off')
    axs[0].set_title('image source')
    axs[0].imshow(x.squeeze())

    axs[1].axis('off')
    axs[1].set_title('image source')
    axs[1].imshow(x_fin.squeeze())

    axs[2].axis('off')
    axs[2].set_title('image destination')
    axs[2].imshow(image_warp)

    plt.savefig('kornia_test.png')
    print('kornia test done.')
    exit()


def test_kornia_affine(x):

    x = x.reshape(-1, *x.shape[-2:])[0:1, None]
    # the source points are the region to crop corners
    mean = 32
    max = 32

    # TODO: Given an unordered set of 2 points. Order them clockwise.
    # points_dst = torch.FloatTensor([[
    #     [20, 20], [50, 20], [50, 50], [20, 50]
    # ]]).to(x.device)

    # Test order
    h, w = 64, 64

    points_test = torch.FloatTensor([[
        [10, 10], [50, 50]
    ]]).to(x.device)
    points_test_2 = torch.FloatTensor([[
        [40, 20], [10, 40]
    ]]).to(x.device)
    points_test = (torch.cat([points_test, points_test_2]) - 32)/32

    bs = points_test.shape[0]
    points_dst, points_src = complete_and_order(points_test)
    print(points_dst)
    print(points_src)
    # Option 2: Hardcoded src points
    # points_src = torch.FloatTensor([[[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]]).to(x.device)
    # points_src = (points_src - mean)/max

    # print(points_src.shape)
    for i in range(points_dst.shape[1]):
        x[:, :, points_dst[0,i,0].long(), points_dst[0,i,1].long()] = 1
    # points_dst = (points_dst - mean)/max

    points_src_aff = points_src[:, 0:3]
    three_zeros = torch.zeros(1, 3, 3).to(x.device).repeat(bs, 1, 1)
    points_src_one = torch.cat([points_src_aff, torch.ones(*points_src_aff.shape[:2], 1).to(x.device)], dim=-1)
    src_x = torch.cat([points_src_one, three_zeros], dim=1)
    src_y = torch.cat([three_zeros, points_src_one], dim=1)
    A = torch.cat([src_x, src_y], dim=-1)

    # the destination points are the image vertexes
    points_dst_aff = points_dst[:, 0:3].permute(0,2,1).reshape(bs, -1, 1)

    # dst_x = points_dst[..., 0:1]
    # dst_y = points_dst[..., 1:2]

    M_aff = torch.bmm(batch_pinv(A, I_factor=1e-20),
                  points_dst_aff
                  ).reshape(bs, 2, 3)

    # compute perspective transform
    # M = kornia.get_perspective_transform(points_src, points_dst)
    # M_inv = kornia.get_perspective_transform(points_dst, points_src)

    grid = F.affine_grid(M_aff,
                         torch.Size((bs, 1, h, w)))
    image_warp_pytorch = F.grid_sample(x.repeat_interleave(bs, dim=0), grid)
    # image_warp_pytorch = kornia.warp_affine(x, M_aff, dsize=(h, w))

    # warp the original image by the found transform
    # img_warp = kornia.warp_perspective(x, M, dsize=(h, w))
    # x_fin= kornia.warp_perspective(img_warp, M_inv, dsize=(h*2, w*2))

    # convert back to numpy
    # image_warp = kornia.tensor_to_image(img_warp[0])

    # create the plot
    fig, axs = plt.subplots(1, 3, figsize=(23, 10))
    axs = axs.ravel()

    axs[0].axis('off')
    axs[0].set_title('image source')
    axs[0].imshow(x.squeeze().cpu())

    # axs[1].axis('off')
    # axs[1].set_title('image source')
    # axs[1].imshow(x_fin.squeeze())
    axs[1].axis('off')
    axs[1].set_title('image destination pytorch')
    axs[1].imshow(image_warp_pytorch[0].squeeze().cpu())

    axs[2].axis('off')
    axs[2].set_title('image destination')
    axs[2].imshow(image_warp_pytorch[1].squeeze().cpu())

    plt.savefig('kornia_test.png')
    print('kornia test done.')
    exit()

def test_final_functions_affine(x):

    x = x.reshape(-1, *x.shape[-2:])[0:1, None]
    # the source points are the region to crop corners
    mean = 32
    max = 32

    # TODO: Given an unordered set of 2 points. Order them clockwise.
    # points_dst = torch.FloatTensor([[
    #     [20, 20], [50, 20], [50, 50], [20, 50]
    # ]]).to(x.device)

    # Test order
    h, w = 64, 64

    points_test = torch.FloatTensor([[
        [10, 10], [50, 50]
    ]]).to(x.device)
    points_test_2 = torch.FloatTensor([[
        [40, 20], [10, 40]
    ]]).to(x.device)
    points_test = (torch.cat([points_test, points_test_2]) - mean)/max
    bs = points_test.shape[0]

    M_center, M_relocate = get_affine_params(points_test)

    image_warp = warp_affine(x.repeat_interleave(bs, dim=0), M_center, h=32, w=32)
    image_ori_warp = warp_affine(image_warp, M_relocate, h=64, w=64)


    # create the plot
    fig, axs = plt.subplots(1, 3, figsize=(23, 10))
    axs = axs.ravel()

    axs[0].axis('off')
    axs[0].set_title('image source')
    axs[0].imshow(x.squeeze().cpu())

    # axs[1].axis('off')
    # axs[1].set_title('image source')
    # axs[1].imshow(x_fin.squeeze())
    axs[1].axis('off')
    axs[1].set_title('image destination pytorch')
    axs[1].imshow(image_warp[0].squeeze().cpu())

    axs[2].axis('off')
    axs[2].set_title('image destination')
    axs[2].imshow(image_ori_warp[0].squeeze().cpu())

    plt.savefig('kornia_test.png')
    print('kornia test done.')
    exit()

def get_affine_params(pts):
    bs, n_pts, xy = pts.shape
    I_factor = 1e-5
    assert n_pts == 2
    assert pts.max() <= 1 and pts.min() >= -1
    # print(pts[0])
    points_dst, points_src = complete_and_order(pts)

    three_zeros = torch.zeros(1, 3, 3).to(pts.device).repeat(bs, 1, 1)
    points_src_one = torch.cat([points_src, torch.ones(*points_src.shape[:2], 1).to(pts.device)], dim=-1)
    src_x = torch.cat([points_src_one, three_zeros], dim=1)
    src_y = torch.cat([three_zeros, points_src_one], dim=1)
    A = torch.cat([src_x, src_y], dim=-1)

    # the destination points are the image vertexes
    points_dst_vec = points_dst.permute(0,2,1).reshape(bs, -1, 1)

    M_center = torch.bmm(batch_pinv(A, I_factor=I_factor),
                      points_dst_vec
                      ).reshape(bs, 2, 3)

    points_src_one = torch.cat([points_dst, torch.ones(*points_dst.shape[:2], 1).to(pts.device)], dim=-1)
    src_x = torch.cat([points_src_one, three_zeros], dim=1)
    src_y = torch.cat([three_zeros, points_src_one], dim=1)
    A = torch.cat([src_x, src_y], dim=-1)

    points_src_vec = points_src.permute(0,2,1).reshape(bs, -1, 1)

    M_relocate = torch.bmm(batch_pinv(A, I_factor=I_factor),
                           points_src_vec
                           ).reshape(bs, 2, 3)

    return M_center, M_relocate

def warp_affine(x, M, h=None, w=None):
    if h is None and w is None:
        h, w = x.shape[-2:]
    ini_shape = x.shape
    if len(x.shape) is not 4:
        x = x.reshape(-1, *x.shape[-3:])
    bs = x.shape[0]
    grid = F.affine_grid(M, torch.Size((bs, 1, h, w)))
    image_warp = F.grid_sample(x, grid, mode='bilinear') #TODO: Check gradients of nearest
    return image_warp.reshape(*ini_shape[:-2], h, w)

def complete_and_order(two_pts):
    '''
    Args:
        two_pts: [b, idx_p, xy]
    Returns:
        three_ordered_pts: [b, 3, xy]

    '''
    bs, _, _ = two_pts.shape

    idx_w_switch = two_pts[:,0,0] > two_pts[:,1,0]
    switch = torch.LongTensor([[1,1], [0,0]]).to(two_pts.device)

    all_idx = torch.zeros_like(two_pts).long()
    all_idx[:, 1] = 1
    all_idx_w = all_idx
    all_idx_h = all_idx

    all_idx_w[idx_w_switch] = switch
    # print(idx_w_switch, all_idx)
    two_ordered_pts = two_pts.gather(dim=1, index=all_idx_w)
    # print(two_pts, two_ordered_pts)


    # Third coordinate (always the one with minimum w)
    third_idx = torch.LongTensor([[1,0], [0,1]]).to(two_pts.device)[None].repeat(bs, 1, 1)
    third_coord = two_ordered_pts.gather(dim=1, index=third_idx)[:, 1:]
    three_pts = torch.cat([two_ordered_pts, third_coord], dim=1)

    # Create src points
    idx_h_switch = two_ordered_pts[:,0,1] > two_ordered_pts[:,1,1]

    src_pts = torch.zeros_like(three_pts)
    # option_1 = torch.FloatTensor([[0, h], [w, 0], [0, 0]]).to(two_pts.device)
    # option_2 = torch.FloatTensor([[0, 0], [w, h], [0, h]]).to(two_pts.device)
    option_1 = torch.FloatTensor([[-1, 1], [1, -1], [-1, -1]]).to(two_pts.device)
    option_2 = torch.FloatTensor([[-1, -1], [1, 1], [-1, 1]]).to(two_pts.device)
    src_pts[:] = option_2
    src_pts[idx_h_switch] = option_1

    return three_pts, src_pts


# @staticmethod
def batch_pinv(x, I_factor):
    """
    :param x: B x N x D (N > D)
    :param I_factor:
    :return:
    """

    B, N, D = x.size()

    if N < D:
        x = torch.transpose(x, 1, 2)
        N, D = D, N
        trans = True
    else:
        trans = False

    x_t = torch.transpose(x, 1, 2)

    use_gpu = torch.cuda.is_available()
    I = torch.eye(D)[None, :, :].repeat(B, 1, 1)
    if use_gpu:
        I = I.to(x.device)

    x_pinv = torch.bmm(
        torch.inverse(torch.bmm(x_t, x) + I_factor * I),
        x_t
    )

    if trans:
        x_pinv = torch.transpose(x_pinv, 1, 2)

    return x_pinv