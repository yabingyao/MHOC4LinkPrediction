using SparseArrays
using CSV
function CalAUC(
    #train::SparseMatrixCSC{Int64,Int64},
    train,
    #test::SparseMatrixCSC{Int64,Int64},
    test,
    sim,
    n::Int64,
)
    #train训练集邻接矩阵
    #test测试集邻接矩阵
    #sim相似性指标矩阵
    #n抽样比较次数
    sim = UpperTriangular(sim - sim .* train)
    #只保留测试集和不存在边集合中的边的相似度
    I = ones(Int64, size(train, 1), size(train, 2))
    #I是元素全为1的矩阵
    non = I - train - test
    non = non - Diagonal(non)
    #得到不存在边集合的邻接矩阵，对角线置为零
    test = UpperTriangular(test)
    non = UpperTriangular(non)
    #分别取测试集和不存在边集合中的上三角矩阵
    #用以取出它们对应的相似度分值
    test = sparse(test)
    non = sparse(non)
    #nnz函数的输入必须是稀疏矩阵
    counts_testlinks = nnz(test)
    counts_nonlinks = nnz(non)
    #获取测试集边数和不存在边数目
    test_rand_index = counts_testlinks * rand(1,n)
    #返回相应数目的随机数，也就是counts_testlinks个随机小数
    for i = 1 : length(test_rand_index)
        test_rand_index[i] = ceil(test_rand_index[i])
        #将下标转换成整数
    end
    #println(test_rand_index)
    test_rand_index = collect(Int64, test_rand_index)
    #println(test_rand_index)
    #随机抽取n条测试边，用于获取对应的相似度分值
    non_rand_index = counts_nonlinks * rand(1,n)
    for j = 1 : length(non_rand_index)
        non_rand_index[j] = ceil(non_rand_index[j])
    end
    #println(non_rand_index)
    non_rand_index = collect(Int64, non_rand_index)
    #println(non_rand_index)
    #随机抽取n条不存在边下标，用于获取对应的相似度分值
    test_sim = sim .* test
    non_sim = sim .* non
    #取测试边和不存在边对应的相似度矩阵
    #test_nz_index = findall(test.==1)
    #获取test中
    #findall函数返回,  A = [1 2 0; 3 4 0],
    #findall(isodd, x),CartesianIndex(1, 1)
    #CartesianIndex(2, 1)
    """
    findall(f::Function, A)
    Return a vector I of the indices or keys of A where f(A[I]) returns true. If
    there are no such elements of A, return an empty array.
    """
    test_sim_data = test_sim[test.==1]'
    non_sim_data = non_sim[non.==1]'
    #获取所有测试边和不存在边的预测值
    #必须和前面的随机下标一致，都是数组，这返回的是向量
    test_rand_scores = test_sim_data[test_rand_index]
    non_rand_scores = non_sim_data[non_rand_index]
    CSV.write("/Users/cty/Documents/实验/linkprediction/$(datasets[j])auc.csv", df1)
    CSV.write(".\\result\\HigherOrder10MI$(datasets[j])auc.csv", df1)
    #得到随机抽取边的相似性分值
    n1 = length(test_rand_scores[test_rand_scores .> non_rand_scores])
    n2 = length(test_rand_scores[test_rand_scores .== non_rand_scores])
    auc = round((n1 + 0.5 * n2)/n,digits=4)
end
