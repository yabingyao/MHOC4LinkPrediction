using SparseArrays
using LinearAlgebra
using DataFrames
using MatrixNetworks
#using LightGraphs
function DivideNetNoCon(
    net::SparseMatrixCSC{Int64,Int64},
    ratioTrain::Float64
    )
    dense_net = Array(net)
    #println(dense_net)
    #获取总边数和测试集边数
    #println("原始网络的边数：",nnz(net))
    count_alllinks = nnz(net)/2
    #println("网络总边数: ",count_alllinks)
    #数据集总边数
    count_testlinks = round(Int,count_alllinks*(1-ratioTrain))
    #println("count_testlinks: ",count_testlinks)
    #测试集边数,round向上取整函数
    B = UpperTriangular(dense_net)
    #获取邻接矩阵的上三角
    A = sparse(B)
    #得到稀疏矩阵
    indextuple = findnz(A)
    #找到上三角中的非零值的下标
    df = DataFrame()
    #定义一个dataframe用来存放linklist
    df[0:1133] = indextuple[1]
    df[0:1133] = indextuple[2]
    df[0:1133] = 0
    #dataframe总共有三列，分别为x，y，是否被抽取
    #0代表没被抽取，1代表被抽取
    #println(df)
    test = zeros(Int64,size(dense_net,1),size(dense_net,2))
    #构造一个与net同型的矩阵，也就是测试集，初始化为全零
    #train = net
    count_samplelink = 0
    #用来计数抽取的边数
    while count_samplelink < count_testlinks
        testlink_index = ceil(Int64,rand() * length(indextuple[1]))
        #println("testlink_index ",testlink_index)
        if (testlink_index == 0) || (df[testlink_index,3] == 1)
            continue
        end
        #如果抽取的行已经被抽过了，那么就重新抽取，或者随机数为0
        #随机抽取一行在dataframe中
        source_node_id = df[testlink_index,1]
        target_node_id = df[testlink_index,2]
        #println("source_node_id: ",source_node_id," target_node_id: ",target_node_id)
        dense_net[source_node_id,target_node_id] = 0
        dense_net[target_node_id,source_node_id] = 0
        #println("dense_net: ",dense_net)
        #train[source_node_id,target_node_id] = 0
        #train[target_node_id,source_node_id] = 0
        #把这一条边从训练集中去掉
        #sparsenet = sparse(dense_net)
        #is_connected(A)
        #换成LightGraphs包的函数需要装成图
        # A: a MatrixNetwork or SparseMatrixCSC class
        if (MatrixNetworks.is_connected(sparse(dense_net)))
            #println("connected!")
            df[testlink_index,3] = 1
            #将此行标记为测试边
            test[source_node_id,target_node_id] = 1
            test[target_node_id,source_node_id] = 1
            #将此条边放入测试集中
            count_samplelink = count_samplelink + 1
            #测试集计数加1
            #println("test: ",test)
        else
            #不连通就恢复
            dense_net[source_node_id,target_node_id] = 1
            dense_net[target_node_id,source_node_id] = 1
            #println("dense_net ",dense_net)
            df[testlink_index,3] = 1
        end
    end
    #println("原始网络的边数：",nnz(net)/2)
    train = dense_net
    #println("测试集边数 " ,nnz(sparse(test))/2)
    #println("训练集边数 " ,nnz(sparse(train))/2)
    #println("df: ",df)
    #test = collect(Int64,test)
    #println("finally train: ",train)
    #println("finally test: ",test)
    #test = sparse(test)
    #train = sparse(train)
    return train,test
end
