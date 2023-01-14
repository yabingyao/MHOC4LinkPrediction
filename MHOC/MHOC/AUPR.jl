using  LinearAlgebra
using SparseArrays
using MLBase
function AUPR(
    train::SparseMatrixCSC{Int64,Int64},
    test::SparseMatrixCSC{Int64,Int64},
    sim,
    n::Int64,
)
    # params n : 就是比较次数
    #sim所有边的相似性矩阵
    #计算AUC，输入计算的相似性矩阵
    #tempsim = sim - sim .* train
    #println("Ok sim ")
    sim = UpperTriangular(sim - sim .* train)
    #测试集和不存在边集合中的边的相似性上三角矩阵
    #println("sim: ",sim)
    A = ones(Int64, size(train, 1), size(train, 2))
    #得到全为1的与训练集矩阵同型的矩阵
    non = A - train - test
    #得到不存在变的矩阵
    #=
    for i = 1:size(non, 1)
        non[i, i] = 0
    end
    =#
    non -= Diagonal(non)
    #去掉对角线元素
    #println("ok nnz")
    test = UpperTriangular(test)
    non = UpperTriangular(non)
    #println("test: ",test)
    #println("non: ",non)
    #分别取测试集合不存在边集合的上三角矩阵，用以去除他们对应的相似度分值
    #println("ok uppertriangluar")
    #println(typeof(test))
    test = sparse(test)
    non = sparse(non)
    test_num = nnz(test)
    #println("测试集变数：",test_num)
    non_num = nnz(non)
    #println("不存在集变数",non_num)
    #nnz返回非零值得个数
    #println("ok")
    test_rd_index = test_num * rand(1, n)
    #获得测试边中的随机得分序列
    #println(typeof(test_rd_index),"测试集得分序列下标为",test_rd_index)

    for w = 1:length(test_rd_index)
        test_rd_index[w] = ceil(test_rd_index[w])
    end
    #对测试集下标向上取整
    test_rd_index = collect(Int64, test_rd_index)
    #collect将小数装换成整数

    #println(typeof(test_rd_index))
    #cld向上取整，n为抽样比较的次数
    non_rd_index = non_num * rand(1, n)
    #println(typeof(non_rd_index),"不存在边集得分序列下标为",non_rd_index)
    for e = 1:length(non_rd_index)
        non_rd_index[e] = ceil(non_rd_index[e])
    end
    non_rd_index = collect(Int64, non_rd_index)
    #获得一系列随机下标换换成成整数将小数
    test_pre = sim .* test
    #print("test_pre:",sparse(test_pre))
    #测试集相似性矩阵

    non_pre = sim .* non
    #print("non_pre:",sparse(non_pre))
    #println(typeof(non_pre),non_pre)
    #获得不存在边的相似性

    #test_data = test_pre(test == 1)’
    test_nz_indices = findall(test.==1)
    #println(test_nz_indices)
    #找到test中等于1的坐标集合
    #就是把测试集中有边的对应坐标的相似性值取出来
    test_sim_data = zeros(Float64,length(test_nz_indices))
    #println("length:test_nz_indices: ",length(test_nz_indices))
    for i = 1 : length(test_nz_indices)
        #print("test_nz_indices: ",test_nz_indices[i])
        #print("test_pre[test_nz_indices[i]]:  ",test_pre[test_nz_indices[i]])
        test_sim_data[i] = test_pre[test_nz_indices[i]]
    end
    #println("test_sim_data:",test_sim_data)
    non_nz_indices = findall(non.==1)
    non_sim_data = zeros(Float64,length(non_nz_indices))
    for j = 1 : length(non_nz_indices)
        non_sim_data[j] = non_pre[non_nz_indices[j]]
    end
    #println("non_sim_data:",non_sim_data)
    test_record = zeros(Float64,length(test_rd_index))
    for a = 1: length(test_rd_index)
        test_record[a] = test_sim_data[test_rd_index[a]]
    end

    non_record = zeros(Float64,length(non_rd_index))
    for b = 1 : length(non_rd_index)
        #print("non_rd_index[b] :",non_rd_index[b])
        #tempvalue = non_rd_index[b]
        #print("non_sim_data: ",non_sim_data[tempvalue])
        non_record[b] = non_sim_data[non_rd_index[b]]
    end

    n1 = length(findall(test_record.>non_record))
    n2 = length(findall(test_record.==non_record))
    #=
    testtuple = findnz(test)
    #获得test中不为零的位置的坐标和值
    I = testtuple[1]
    #行
    println("行",I)
    J = testtuple[2]
    #列
    println("列",J)
    #println("no")
    #println(test_num)
    test_data = Array{Int64}(undef, test_num)
    println("测试集数据集合：",typeof(test_data),test_data)
    for j = 1:test_num
        test_data = test_pre[I[j],J[j]]
    end
    #non_data = non_pre(non == 1)’
    #println("daolemei")
    nontuple = findnz(non)
    M = nontuple[1]
    N = nontuple[2]
    println(non_num)
    non_data = Array{Int64}(undef, non_num)
    for k = 1:non_num
        non_data = non_pre[M[k], N[k]]
    end
    println("bounderror")
    #test_rd = test_data(test_rd)
    test_record = Array{Float64}(undef, length(test_rd_index))
    #根据得分序列获得测试边中的随机得分
    for m in test_rd_index
        test_record = test_data[m]
    end
    println(length(test_data))
    println(length(test_rd_index))
    for a = 1:length(test_rd_index)
        test_record = test_data[a]
    end
    println("bounderror")
    #non_rd = non_data(non_rd)
    non_record = Array{Float64}(undef, length(non_rd_index))
    for n in non_rd_index
        non_record = non_data[n]
    end
    #n1 = length(find(test_rd > non_rd))
    =#

    #=
    n1, n2 = 0
    q = 1
    while (q < length(test_record))
        if (test_record[q] > non_record[q])
            n1 += 1
        elseif (test_record[q] == non_record)
            n2 += 1
        end
        q += 1
    end
    =#
    #n2 = length(find(test_rd == non_rd))
    auc = (n1 + 0.5 * n2) / n
    return auc
end
