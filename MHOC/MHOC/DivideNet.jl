using LightGraphs
using SparseArrays
using LinearAlgebra
function DivideNet(net::SparseMatrixCSC{Int64,Int64},ratioTrain::Float64)
    #划分训练集和测试集，保证训练集连通,ratioTrain训练集比例
    edges = nnz(net)
    #println("总边数",edges/2)
    num_testlinks = Int64(round((1-ratioTrain) * (edges/2)))
    #println("计划测试集边数: ",num_testlinks)
    println("进入划分训练集")
    #得到网络中的测试集的边数目
    #nnz返回非零值得个数
    #findnz(A)
    # Return a tuple (I, J, V) where I and J are the row and column indices of the stored ("structurally non-zero") values in sparse matrix A,
    # and V is a vector of the values.
    #julia> A = sparse([1 2 0; 0 0 3; 0 4 0])
    #3×3 SparseMatrixCSC{Int64,Int64} with 4 stored entries:
    #[1, 1]  =  1
    #[1, 2]  =  2
    #[3, 2]  =  4
    #[2, 3]  =  3

    #julia> findnz(A)
    #([1, 1, 3, 2], [1, 2, 2, 3], [1, 2, 4, 3])
    B = UpperTriangular(net)
    #得到上三角矩阵
    A = sparse(B)
    #转换成稀疏矩阵
    indextuple = findnz(A)
    #找到上三角矩阵中的非零值，也就是边
    #flag = indextuple[3]
    #flag作为选边的标志，0代表此条边已经选过，1代表此条边没选
    column1 = indextuple[1]
    column2 = indextuple[2]
    #把矩阵中非零元素的行列记录下来
    linklist=Array{Int16}(undef,length(column1),2)
    #坐标元素拼成一个nx2的矩阵，第一列是行，第二列是列
    for i =1:length(column1)
        linklist[i,1]=column1[i]
        linklist[i,2]=column2[i]
    end
    test = spzeros(size(net,1),size(net,2))
    #println(nnz(test))
    #构造一个与net同型的矩阵，也就是测试集，初始化为全零
    test_edges = 0
    while( test_edges < num_testlinks)
        #随机选择一条边
        index_link =Int64(round(rand()*length(column1)))
        #println(index_link)
        #julia中数组下标必须从1开始
        if(index_link==0)
            index_link=1
        end
        uid1 = linklist[index_link,1]
        uid2 = linklist[index_link,2]
        #println(uid1)
        #println(uid2)
        net[uid1,uid2]= 0
        net[uid2,uid1]= 0
        if(is_connected(SimpleGraph(net)))
            test[uid1,uid2]=1
            test[uid2,uid1]=1
            test_edges += 1
        else
            net[uid1,uid2]=1
            net[uid2,uid1]=1
        end
    end
    train = net
    #println("训练集边数： ",nnz(train)/2)
    test = collect(Int64,test)
    test = sparse(test)
    #println("测试集边数: ",nnz(test)/2)
    return train,test
end
