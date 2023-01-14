include("HigherOrderClustering.jl")
include("read_undir_txt.jl")
using DataFrames
datasets = ["CAG_mat_364_10474.txt","c-fat_500_9139.txt","130bit_584_6058.txt","moreno_names_names_1707_9059.txt","socfb-Caltech_762_16651.txt","socfb-Reed98_962_18812.txt","SciMet_2678_10368.txt","progas_1900_8797.txt","D_816_815_7526.txt","fb-forum_899_7036.txt","foodweb-baydry_128_2106.txt"]
    #计算每个网络的高阶聚类系数
ccfs2 = Float64[]
ccfs3 = Float64[]
ccfs4 = Float64[]
for i = 1:length(datasets)
    data = load_example_data(datasets[i])
    ccfs2all = higher_order_ccfs(data,2)
    push!(ccfs2,ccfs2all.avg_hoccf)
    ccfs3all = higher_order_ccfs(data, 3)
    push!(ccfs3,ccfs3all.avg_hoccf)
    ccfs4all = higher_order_ccfs(data, 4)
    push!(ccfs4,ccfs4all.avg_hoccf)
end
df = DataFrame()
df.dataset = datasets
df.hoccfs2 = ccfs2
df.hoccfs3 = ccfs3
df.hoccfs4 = ccfs4
print(df)
