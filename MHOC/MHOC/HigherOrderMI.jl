using SparseArrays
using Combinatorics
using LightGraphs
include("HigherOrderClustering.jl")
"""
struct hoccf_data
    order::Int64
    global_hoccf::Float64
    avg_hoccf::Float64
    avg_hoccf2::Float64
    local_hoccfs::Vector{Float64}
    ho_wedge_counts::Vector{Int64}
    clique_counts::Vector{Int64}
end
hoccf_data
----------

This is the data type returned by the function that computes higher-order clustering coefficients.

The field values are

order::Int64
    The order of the clustering coefficient (order = 2 is the classical
    clustering coefficient definition).

global_hoccf::Float64
    The global higher-order clustering coefficient.

avg_hoccf::Float64
    The average higher-order clustering coefficient (the mean is taken over
    nodes at the center of at least one wedge).

avg_hoccf2::Float64
     The average higher-order clustering coefficient, where the local clustering
    of a node not in any wedge is considered to be 0.

local_hoccfs::Vector{Float64}
    The vector of local higher-order clustering coefficients. If a node is not
    the center of at least one wedge, then the value is 0.

ho_wedge_counts::Vector{Int64}
    Higher-order wedge counts of nodes: ho_wedge_counts[v] is the number of higher-order
    wedges with node v at its center.

clique_counts::Vector{Int64}
    Clique counts of nodes: clique_counts[v] is the number of k-cliques containing
    node v where k = order + 1.
"""
function CalHigherOrderMI(adjacent_matrix,order)
    g = SimpleGraph(adjacent_matrix)
    rows = nv(g)
    columns = rows
    mutual_information = zeros(Float64,rows,columns)
    col_degree=degree(g)
    #存放每一个节点的度值
    ccfs2 = clustercoeffs(sparse(adjacent_matrix))
    ccfs3all = higher_order_ccfs(sparse(adjacent_matrix), 3)
    ccfs3 = ccfs3all.local_hoccfs
    ccfs4all = higher_order_ccfs(sparse(adjacent_matrix), 4)
    ccfs4 = ccfs4all.local_hoccfs
    #得到网络的3阶聚类系数，这需要的是局部聚类系数也就是
    #local_clustering = clustercoeffs(sparse(adjacent_matrix))
    #计算每一个节点的局部聚类系数
    #利用matrixnetworks库得clustercoeffs函数
    #A = load_matrix_network("clique-10")
    #cc = clustercoeffs(MatrixNetwork(A))
    #local_hoccfs::Vector{Float64}
    function ccfs(node::Int64,order::Int64)
        #计算网络中一个节点的局部高阶聚类系数
        #融合二阶和三阶局部聚类系数
        if order==3
            return ccfs3[node]
            #只返回三阶聚类系数
        elseif order==4
            return ccfs4[node]
            #只返回四阶聚类系数
        elseif order==23
            return 1 - (1-ccfs2[node])*(1-ccfs3[node])
            #融合二阶和三阶聚类系数
        else
            return 1 - (1-ccfs2[node])*(1-ccfs3[node])*(1-ccfs4[node])
            #同时融合二阶三阶和四阶
        end
    end
    edges = ne(g)
    #计算网络的连边数目
    information_entropy_xy = zeros(Float64,rows,columns)
    #P(Lmn1) = (Combine(M,Km) - Combine(M-Kn,Km))/Combine(M,Km)
    function Probability_mn(M::Int64,Km::Int64,Kn::Int64)
        temp = big(col_degree[Kn])
        #println("M ",M," Km ",Km," Kn ",Kn," col_degree[Kn] ",col_degree[Kn]," col_degree[Km] ",col_degree[Km] )
        a = binomial(big(M),temp)
        #println("Combine(M,col_degree[Kn])",Combine(M,col_degree[Kn]))
        b = binomial(big(M-col_degree[Km]),temp)
        #println("Combine(M-Km,col_degree[Kn])",Combine(M-Km,col_degree[Kn]))
        result = 1 - b/a
        #println(result)
        return result
    end
    #I(lmn1) = -log2(Pmn1)
    for i = 1:rows
        for j = 1:columns
            if (i!=j&&i<j)
                #是对称阵，只需要算上三角information_entropy_xy[i,j] = information_entropy_xy[j,i]
                information_entropy_xy[i,j] = round(-log2(Probability_mn(edges,i,j)),digits=4)
                #最终结果保留四位小数
            end
        end
    end
    #println(information_entropy_xy)
    #从节点z的邻居集合中枚举所有的边，然后计算I(Lmn1)
    #I(Lmn1;z) = I(Lmn1)-I(Lmn1|z)
    #I(Lxy1;z) = 1/z*(z-1)segema(m!=n)m,n∈(z)I(Lmn1;z)
    #就是以z为中心的所有邻居的信息集合
    #先找出z的所有邻居，在计算平均的信息熵
    #node_neighbors_set = zeros(Int64,rows,columns)
    #用来存放每一个节点的邻居，存放的是他的节点编号(1,3)=3说明节点1和节点3是邻居
    """
    for m = 1 : rows
        for n = 1 : columns
            if(adjacent_matrix[m,n] != 0)
                node_neighbors_set[m,n]=n
            end
        end
    end
    """
    neighbors_probability = zeros(Float64,rows)
    #用来存放每一个节点的所有邻居信息
    neighbors_pairs = zeros(Int64,rows)
    #存放枚举出来的节点对，也就是边
    for m = 1 : rows
        neighbors_pairs = collect(combinations(neighbors(g,m),2))
        #得到此节点的邻居组合
        temp_result = 0
        #segema(m!=n)m,n∈(z)I(Lmn1;z)
        for n = 1:length(neighbors_pairs)
            #if(neighbors_pairs[n][1]!=0 && neighbors_pairs[n][2]!= 0)
            x = neighbors_pairs[n][1]
            y = neighbors_pairs[n][2]
                #I(Lmn1;z) = I(Lmn1)-I(Lmn1|z)
            tempvalue = ccfs(m,order)
            if tempvalue == 0
                result = 0
            else
                result = log2(tempvalue)
            end
            temp_result += information_entropy_xy[x,y] + result
            #end
        end
        #I(Lxy1;z) = 1/z*(z-1)segema(m!=n)m,n∈(z)I(Lmn1;z)
        neighbors_probability[m] = 1/(col_degree[m] * (col_degree[m]-1)) * temp_result
        #度值出现1就出问题
    end

    #Sxy(MI) = -I(Lxxy1|Oxy)
    #I(Lxy1|Oxy) = I(Lxy1) - I(Lxy1;Oxy)
    #Sxy = -I(Lxxy1|Oxy) = segema(z∈ Oxy)I(Lxy1;z) - I(Lxy1)
    for i = 1:rows,j = 1:columns#少算了节点x和节点y的其他共同邻居
        if (i!=j&&i<j&&adjacent_matrix[i,j]==0)
            #只求上三角a,b的共同邻居与b,a一样,并且只算没有连边的节点对的sim
            temp_mutual_result = 0
            com_neigh = common_neighbors(g,i,j)
            #segema(z∈ Oxy)I(Lxy1;z)
            for m in com_neigh
                #if(com_neigh[m] != 0)
                #共同邻居经过过滤之后没有0
                temp_mutual_result += neighbors_probability[m]
                #print(mutual_information[i,j]," ")
                #end
            end
            #segema(z∈ Oxy)I(Lxy1;z) - I(Lxy1)
            mutual_information[i,j] = temp_mutual_result - information_entropy_xy[i,j]
            #print("总的",mutual_information[i,j]," ")
        end
    end
    for i = 1:rows,j=1:columns
        if(isinf(mutual_information[i,j]))
            mutual_information[i,j]=0
        end
    end
    for i = 1:rows,j=1:columns
        if(isnan(mutual_information[i,j]))
            mutual_information[i,j]=0
        end
    end
    return mutual_information
end
