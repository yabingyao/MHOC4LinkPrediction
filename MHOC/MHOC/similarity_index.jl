using SparseArrays
using LinearAlgebra
using LightGraphs
#using MatrixNetworks
#function CN(adjacent_matrix::SparseMatrixCSC{Int64,Int64})
function CN(adjacent_matrix)
    #计算CN指标,adjacent_matrix是网络的邻接矩阵,sim是相似性矩阵
    #Sxy = |T(x)andT(y)|,T(x)为x的邻居
    #sim = adjacent_matrix * adjacent_matrix
    #return sim
    g = SimpleGraph(adjacent_matrix)
    rows = nv(g)
    sim = zeros(Int64,rows,rows)
    for i = 1:rows,j=1:rows
        if i!=j && i<j && adjacent_matrix[i,j]==0
            sim[i,j] = length(common_neighbors(g,i,j))
        end
    end
    return sim
end
"""
function AA(adjacent_matrix::SparseMatrixCSC{Int64,Int64})
    #计算每一个度的log值，Julia中log函数只能作用于方阵
    tempvec = [log(x) for x in col_degree]
    #对每一个度值计算对数值
    #这是Julia中的推导式[f(x,y,...) for x=rx, y=ry,...]
    temp_adj_m = adjacent_matrix ./ repeat(tempvec,size(adjacent_matrix,1),1)
    for i=1:size(adjacent_matrix,1),j=1:size(adjacent_matrix,2)
        if(isinf(adjacent_matrix[i,j]))
            adjacent_matrix[i,j]=0
        end
    end
    for i=1:size(adjacent_matrix,1),j=1:size(adjacent_matrix,2)
        if(isnan(adjacent_matrix[i,j]))
            adjacent_matrix[i,j]=0
        end
    end
    for i = 1:size(temp_adj_m,1),j=1:size(temp_adj_m,2)
        if(isinf(temp_adj_m[i,j]))
            temp_adj_m[i,j]=0
        end
    end
    for i = 1:size(temp_adj_m,1),j=1:size(temp_adj_m,2)
        if(isnan(temp_adj_m[i,j]))
            temp_adj_m[i,j]=0
        end
    end
    sim = adjacent_matrix * temp_adj_m
    return sim
end
"""
function AA(adjacent_matrix)
    #计算AA指标,Sxy = segema(1/log(Kz)),z属于x和y的共同邻居，Kz为节点度
    g = SimpleGraph(adjacent_matrix)
    #将邻接矩阵构造成一个简单图
    col_degree=degree(g)
    rows = nv(g)
    columns = rows
    sim = zeros(Float64,rows,columns)
    for i=1:rows,j=1:columns
        if i!=j && i<j && adjacent_matrix[i,j]==0
            tempvec =  [1/log2(col_degree[node]) for node in common_neighbors(g,i,j)]
            for x in tempvec
                if isinf(x)||isnan(x)
                    continue
                else
                    sim[i,j] += x
                end
            end
        end
    end
    return sim
end

#function RA(adjacent_matrix::SparseMatrixCSC{Int64,Int64})
"""
function RA(adjacent_matrix)
    #计算RA指标 Sxy = segema(1/Kz),z是x和y的共同邻居
    temp_adjm = adjacent_matrix ./ repeat(sum(adjacent_matrix,dims =1),size(adjacent_matrix,1),1)
    #计算每个节点的权重，1/k_i
    #println(typeof(temp_adjm))
    for i = 1:size(adjacent_matrix,1),j=1:size(adjacent_matrix,2)
        if(isinf(adjacent_matrix[i,j]))
            adjacent_matrix[i,j]=0
        end
    end
    for i=1:size(adjacent_matrix,1),j=1:size(adjacent_matrix,2)
        if(isnan(adjacent_matrix[i,j]))
            adjacent_matrix[i,j]=0
        end
    end
    sim = adjacent_matrix * temp_adjm
    #println(typeof(sim))
    return sim
end
"""
function RA(adjacent_matrix)
    #计算RA指标 Sxy = segema(1/Kz),z是x和y的共同邻居
    g = SimpleGraph(adjacent_matrix)
    #将邻接矩阵构造成一个简单图
    col_degree=degree(g)
    rows = nv(g)
    columns = rows
    sim = zeros(Float64,rows,columns)
    for i=1:rows,j=1:columns
        if i!=j && i<j && adjacent_matrix[i,j]==0
            tempvec =  [1/col_degree[node] for node in common_neighbors(g,i,j)]
            for x in tempvec
                if isinf(x)||isnan(x)
                    continue
                else
                    sim[i,j] += x
                end
            end
        end
    end
    return sim
end

#function PA(adjacent_matrix::SparseMatrixCSC{Int64,Int64})
"""
function PA(adjacent_matrix)
    #计算PA指标 Sxy = Kx*Ky
    degree = sum(adjacent_matrix,dims=2)
    #sum(adjacent_matrix,dims=2),计算矩阵的行和返回列向量，也就是度,sum(A,dims=1)计算每一列的和
    sim = degree * degree'
    sim = SparseMatrixCSC(sim)
    # A' 就是矩阵A的矩阵转置
    return sim
end
"""
function PA(adjacent_matrix)
    #计算PA指标 Sxy = Kx*Ky
    g = SimpleGraph(adjacent_matrix)
    #将邻接矩阵构造成一个简单图
    col_degree=degree(g)
    rows = nv(g)
    columns = rows
    sim = zeros(Float64,rows,columns)
    for i=1:rows,j=1:columns
        if i!=j && i<j && adjacent_matrix[i,j]==0
            sim[i,j] = col_degree[i] * col_degree[j]
        end
    end
    return sim
end
function LHN1(adjacent_matrix::SparseMatrixCSC{Int64,Int64})
    #计算LHN1指标
    sim = adjacent_matrix * adjacent_matrix
    #完成分子的计算，分子同CN
    degree = sum(adjacent_matrix,dims = 2)
    degree = degree * degree'
    #完成分母的计算
    sim = sim ./ degree
    for i = 1:size(sim,1),j=1:size(sim,2)
        if(isinf(sim[i,j]))
            sim[i,j]=0
        end
    end
    for i=1:size(adjacent_matrix,1),j=1:size(adjacent_matrix,2)
        if(isnan(adjacent_matrix[i,j]))
            adjacent_matrix[i,j]=0
        end
    end
    return sim
end

function Jaccard(adjacent_matrix::SparseMatrixCSC{Int64,Int64})
    #计算jaccard指标,Sxy = |T(x)andT(y)|/|T(x)orT(y)|
    sim = adjacent_matrix * adjacent_matrix
    degree = sum(adjacent_matrix,dims=1)
    #sum(matrix,dim)如果求行和，返回列向量；求列和返回行向量
    #repeat(vector,x,y),vector是行向量
    tempmatix = repeat(degree,size(adjacent_matrix,1),1)
    #
    tempmatix = tempmatix + tempmatix'
    tempmatix = tempmatix - sim
    sim =  sim ./ tempmatix
    for i = 1:size(sim,1),j=1:size(sim,2)
        if(isinf(sim[i,j]))
            sim[i,j]=0
        end
    end
    for i=1:size(adjacent_matrix,1),j=1:size(adjacent_matrix,2)
        if(isnan(adjacent_matrix[i,j]))
            adjacent_matrix[i,j]=0
        end
    end
    return sim
end

function LP(adjacent_matrix::SparseMatrixCSC{Int64,Int64},lambda::Float64)
    #计算LP指标
    sim = adjacent_matrix * adjacent_matrix
    #二阶路径
    sim = sim + lambda * (sim * adjacent_matrix)
    #二阶路径 + 参数 * 三阶路径
    return sim
end

function Katz(adjacent_matrix,lambda::Float64)
    #计算katz指标
    sim = I(size(adjacent_matrix,1)) - lambda * adjacent_matrix
    sim1 = inv(sim)
    # inv求矩阵的逆，I求对应的维数的单位阵 sparse求稀疏阵
    sim = sim1 - sparse(I(size(adjacent_matrix,1)))
    return sim
end
