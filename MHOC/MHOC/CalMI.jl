using SparseArrays
using Combinatorics
using LightGraphs
using MatrixNetworks
#function MI(adjacent_matrix::SparseMatrixCSC{Int64,Int64})
function MI(adjacent_matrix)
    #最终得到的相似性矩阵是对称阵，所以只算上三角矩阵
    g = SimpleGraph(adjacent_matrix)
    rows = nv(g)
    columns = rows
    edges = ne(g)
    #计算网络的连边数目
    mutual_information = zeros(Float64,rows,columns)
    #col_degree=sum(adjacent_matrix,dims=1)
    col_degree = degree(g)
    #存放每一个节点的度值
    local_clustering = clustercoeffs(sparse(adjacent_matrix))
    #计算每一个节点的局部聚类系数
    #利用matrixnetworks库得clustercoeffs函数
    #A = load_matrix_network("clique-10")
    #cc = clustercoeffs(MatrixNetwork(A))
    """
    adjacency_lists = Vector{Vector{Int64}}()
    for v = 1:rows
        push!(adjacency_lists, filter(x -> x != v, findnz(A[:, v])[1]))
    end
    #adjacency_lists用来存放每一个节点的邻居集合，是一个二维数组
    """
    """
    function Combine(a::Int64,b::Int64)#求组合数的函数
        n = Float64(a)
        m = Float64(b)
        if m < n - m
            m = n - m
        end
        result = 1
        #println(result)
        for i = m + 1 : n
            result *= i
            #println(result)
        end
        for j = 1 : n - m
            result /= j
        end
        return result
    end
    """
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
            if (i!=j&&i<j&&adjacent_matrix[i,j]==0)
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
            x = neighbors_pairs[n][1]
            y = neighbors_pairs[n][2]
            #I(Lmn1;z) = I(Lmn1)-I(Lmn1|z)
            localcc = local_clustering[m]
            if localcc == 0
                result = 0
            else
                result = log2(localcc)
            end
            temp_result += information_entropy_xy[x,y] + result
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
            #a = Set(adjacency_lists[i])
            #b = Set(adjacency_lists[j])
            #com_neigh = intersect(a,b)
            com_neigh = common_neighbors(g,i,j)
            #common_neighbors是LightGraphs包下的一个函数，用来求共同邻居
            #segema(z∈ Oxy)I(Lxy1;z)
            for m in com_neigh
                #if(com_neigh[m] != 0)
                #共同邻居经过过滤之后没有0
                temp_mutual_result += neighbors_probability[m]
                #print(mutual_information[i,j]," ")
                #end
            end
            #segema(z∈ Oxy)I(Lxy1;z) - I(Lxy1)
            mutual_information[i,j] = round(temp_mutual_result - information_entropy_xy[i,j],digits=4)
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
    #CSV.write("mi.txt",mutual_information)
    return mutual_information
end
