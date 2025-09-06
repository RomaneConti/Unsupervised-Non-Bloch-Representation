
import torch






def CrossProduct(Vec1, Vec2):
    """
    Cross product of two 2D vectors.

    Params:
        Vec1/2 (torch.tensor): The first/secound 2D vector.
            Should be a 1D tensor with two elements, 

    Returns:
        (torch.tensor): The cross product. Calculated as x1*y2 - x2*y1.
    """

    return Vec1[0]*Vec2[1] - Vec1[1]*Vec2[0]






def BoxTest(A, B, C, D):
    """
    A quick rejection test test to check if two segments (AB and CD) can intersect.

    Params:
        A/B (torch.tensor): The end point of vector (segment) AB.
            Should be a 1D tensor with two elements.
        C/D (torch.tensor): The end point of vector (segment) CD.
            Should be a 1D tensor with two elements.

    Returns:
        (bool): True if  AB and CD may intersect (bounding box have overlap), False otherwise.

    Examples:
        >>> A = torch.tensor([0, 0])
        >>> B = torch.tensor([1, 1])
        >>> C = torch.tensor([1, 0])
        >>> D = torch.tensor([0, 1])
        >>> BoxTest(A, B, C, D)
        True
    """

    return not(max(C[0], D[0])<min(A[0], B[0]) or max(A[0], B[0])<min(C[0], D[0]) or 
               max(C[1], D[1])<min(A[1], B[1]) or max(A[1], B[1])<min(C[1], D[1]))

 




def CrossCheck(A, B, C, D):
    """
    Determine the type of intersection of two vectors (AB and CD).

    The intersections are classified into three types:
    - Type 0: No intersection.
    - Type 1: Intersection exists and lines are not parallel.
        This type includes cases that one end-point of a segment locate just on another segment.
    - Type 2: Intersection exists and lines are parallel.

    Params:
        A/B (torch.tensor): The end point of vector (segment) AB.
            Should be a 1D tensor with two elements.
        C/D (torch.tensor): The end point of vector (segment) CD.
            Should be a 1D tensor with two elements.

    Returns:
        type (int): Type of intersection (0, 1, or 2).

    Examples:
        >>> A = torch.tensor([0., 0.])
        >>> B = torch.tensor([1., 0.])
        >>> C = torch.tensor([0.5, 0.])
        >>> D = torch.tensor([1.5, 0.])
        >>> E = torch.tensor([0.5, 1.])
        >>> CrossCheck(A, B, C, D)
        2

        >>> CrossCheck(A, B, C, E)
        1
    """

    AB, AC, AD = B-A, C-A, D-A
    CD, CA, CB = D-C, A-C, B-C

    cross_AB_CD = CrossProduct(AB, CD)
    cross_AB_AC = CrossProduct(AB, AC)
    cross_AB_AD = CrossProduct(AB, AD)
    cross_CD_CA = CrossProduct(CD, CA)
    cross_CD_CB = CrossProduct(CD, CB)
    cond1 = (cross_AB_AC * cross_AB_AD < 0) or ((cross_AB_AC==0) ^ (cross_AB_AD==0))
    cond2 = (cross_CD_CA * cross_CD_CB < 0) or ((cross_CD_CA==0) ^ (cross_CD_CB==0))

    if cond1 and cond2:
        type = 1
    elif (cross_AB_CD == 0) and (cross_AB_AC == 0 and BoxTest(A, B, C, D)):
        type = 2
    else:
        type = 0

    return type






def Intersection(A, B, C, D):
    """
    Intersection point of two 2D line segments AB and CD.

    The intersection is got by solving the equation of two lines
        a1*x + b1*y + c1 = 0
        a2*x + b2*y + c2 * 0

    Params:
        A/B (torch.tensor): The end point of vector (segment) AB.
            Should be a 1D tensor with two elements.
        C/D (torch.tensor): The end point of vector (segment) CD.
            Should be a 1D tensor with two elements.

    Returns:
        torch.tensor: Intersection point (X, Y) of segments AB and CD, represented as a 1D tensor.

    Notes:

    Examples:
        >>> A = torch.tensor([0., 0.])
        >>> B = torch.tensor([1., 1.])
        >>> C = torch.tensor([1., 0.])
        >>> D = torch.tensor([0., 1.])
        >>> Intersection(A, B, C, D)
        tensor([0.5000, 0.5000])
    """

    a1, b1, a2, b2 = B[1]-A[1], A[0]-B[0], D[1]-C[1], C[0]-D[0]
    c1, c2 = A[0]*a1 + A[1]*b1, C[0]*a2 + C[1]*b2

    X = (b2*c1 - b1*c2) / (a1*b2 - a2*b1)
    Y = (a1*c2 - a2*c1) / (a1*b2 - a2*b1)

    return torch.tensor([X, Y])






def PolyArea_SimplePoly(polygon):
    """
    Area of a 2D simple polygon (non-self-intersected polygon)
    S = sum x_n*y_n+1 + x_n+1*y_n

    Params:
        polygon (torch.tensor): A N*2 tensor represecting the vertices of the polygon.
            The jth vertex is given by polygon[j]

    Returns:
        (torch.tensor): area of the simple polygon

    Examples:
        >>> polygon1 = torch.tensor([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
        >>> polygon2 = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
        >>> PolyArea_SimplePoly(polygon1)
        tensor(2.)

        >>> PolyArea_SimplePoly(polygon2)
        tensor(0.)
    """

    assert polygon.dim() == 2
    assert polygon.size(1) == 2

    return torch.abs(torch.sum(polygon[:, 0] * (polygon[:, 1].roll(-1)) - 
                               polygon[:, 0].roll(-1) * polygon[:, 1]))






def PolyArea(polygon):
    """
    Area of a polygon (may be self-intersected)
    Once meets an self-intersected point, partition the polygon into two parts
    - For type 1 intersection
        Suppose Edge(i, i+1) and Edge(j, j+1) (i<j) intersect at P
        We partition the polygon into
            polygon1: ... → V(i) → P → V(j+1) → ...
            polygon2: P → V(i+1) → ... → V(j) → P
    - For type 2 intersection
        Suppose Edge(i, i+1) and Edge(j, j+1) (i<j) are overlapped
        We partition the polygon into
            polygon1: ... → V(i) → V(j+1) → ...
            polygon2: V(i+1) → ... → V(j)

    Params:
        polygon (torch.tensor): A tensor representing the vertices of the polygon in 2D space.
            Each vertex is represented as a 1D tensor of two elements (x, y).

    Returns:
        (torch.tensor): The area of the polygon.

    Examples:
        >>> polygon1 = torch.tensor([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
        >>> polygon2 = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
        >>> PolyArea(polygon1)
        tensor(2.)

        >>> PolyArea(polygon2)
        tensor(1.)
    """
    
    assert polygon.dim() == 2
    assert polygon.size(1) == 2

    L = len(polygon)
    if L <= 3:
        area = PolyArea_SimplePoly(polygon)
    
    else:
        # the start indices for finding the intersection 
        search_lst = [(id1, id2) for id1 in range(L-1) for id2 in range(id1+2, L)]
        search_lst = search_lst[ :L-3] + search_lst[L-2: ]
        
        # search intersections or overlapping edges
        intersect_type = 0
        search_id = 0
        while intersect_type == 0 and search_id < len(search_lst):
            id1, id2 = search_lst[search_id]
            # intersect_type = CrossCheck(polygon[id1], polygon[(id1+1)%L], 
            #                             polygon[id2], polygon[(id2+1)%L])
            if BoxTest(polygon[id1], polygon[(id1+1)%L], polygon[id2], polygon[(id2+1)%L]):
                intersect_type = CrossCheck(polygon[id1], polygon[(id1+1)%L], 
                                            polygon[id2], polygon[(id2+1)%L])
            # print([intersect_type, id1, id2, search_id])
            search_id += 1

        # if have intesection, partition the polygon
        match intersect_type:
            case 0:
                area = PolyArea_SimplePoly(polygon)
            
            case 1:
                id1, id2 = search_lst[search_id-1]
                P = Intersection(polygon[id1], polygon[id1+1], 
                                 polygon[id2], polygon[(id2+1)%L])
                
                # note that only id2+1 may out of range
                polygon1 = torch.cat((polygon[ : id1+1],
                                      P.unsqueeze(0),
                                      polygon[id2+1 : ]), dim=0)
                polygon2 = torch.cat((polygon[id1+1 : id2+1],
                                      P.unsqueeze(0)), dim=0)
                
                area = PolyArea(polygon1) + PolyArea(polygon2)

            case 2:
                id1, id2 = search_lst[search_id-1]

                polygon1 = torch.cat((polygon[ : id1+1],
                                      polygon[id2+1 : ]), dim=0)
                polygon2 = polygon[id1+1 : id2+1]

                area = PolyArea(polygon1) + PolyArea(polygon2)

    return area