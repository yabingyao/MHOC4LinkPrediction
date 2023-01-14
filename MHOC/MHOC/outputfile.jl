using DataFrames
using CSV

include("read_undir_txt.jl")
include("similarity_index.jl")
include("DivideNet_notconnected.jl")
include("AUC.jl")
include("Calaupr.jl")
#include("CalNSI.jl")
#datasets = ["USAir.txt","NS.txt","PB.txt","Yeast.txt","Celegans.txt","FWFB.txt","Power.txt","Router.txt"]
#datasets = ["foodweb-baydry_128_2106.txt","USAir_332_2126.txt","NetScience_379_914.txt","celegans_297_2148.txt","arenas-email_1133_5451.txt","Yeast_2375_23386.txt","blogs_1222_16714.txt","power_4941_13188.txt","Router_5022_12516.txt"
#datasets = ["foodweb-baydry_128_2106.txt","USAir_332_2126.txt","NetScience_379_914.txt","celegans_297_2148.txt"]
#datasets = ["Yeast_2375_23386.txt","blogs_1222_16714.txt","power_4941_13188.txt","Router_5022_12516.txt"]
#datasets = ["Karate_34_78.txt"]
datasets = ["blogs_1222_16714.txt"]
#clique_counts = zeros(Int64, size(A, 1))
#simarray = ["CN" "AA" "RA" "PA" "LHN1" "Jaccard" "LP"]

counts = 100
#用来存储每一次的auc指标
cnauc = zeros(Float64,counts)
aaauc = zeros(Float64,counts)
raauc = zeros(Float64,counts)
paauc = zeros(Float64,counts)
"""
paauc = zeros(Float64,counts)
lhn1auc = zeros(Float64,counts)
jaccardauc = zeros(Float64,counts)
lpauc = zeros(Float64,counts)
"""
#miauc = zeros(Float64,counts)
#用来存储aupr指标
cnpre = zeros(Float64,counts)
aapre = zeros(Float64,counts)
rapre = zeros(Float64,counts)
papre = zeros(Float64,counts)
"""
papre = zeros(Float64,counts)
lhn1pre = zeros(Float64,counts)
jaccardpre = zeros(Float64,counts)
lppre = zeros(Float64,counts)
"""
#mipre = zeros(Float64,counts)
#auc = zeros(Float64,counts)
#precison = zeros(Float64,counts)

TopL = 100
segama = 0.0001
n = 670000
df1 = DataFrame()
df2 = DataFrame()
for j = 1 : length(datasets)
    println("正在加载数据集",datasets[j])
    data = load_example_data(datasets[j])
    println(datasets[j],"数据集加载完成")
    for i = 1 : counts
        println("第",i,"次")
        println("开始划分训练集和测试集")
        net = DivideNetNoCon(data,0.9)
        println("训练集划分结束")
        train = net[1]
        test = net[2]
        #println("原始网络的边数：",nnz(data)/2)
        #println("训练集网络的边数：",nnz(train)/2)
        #println("测试集网络的边数: ",nnz(test)/2)
        println("开始计算相似性指标")
        cnsim = CN(train)
        println("cn相似性指标计算结束")
        aasim = AA(train)
        println("aa相似性指标计算结束")
        rasim = RA(train)
        println("ra相似性指标计算结束")
        pasim = PA(train)
        println("pa相似性指标计算结束")
        """
        pasim = PA(train)
        lhn1sim = LHN1(train)
        jaccardsim = Jaccard(train)
        lpsim = LP(train,segama)
        """
        #println("mi相似性指标计算开始")
        #misim = MI(train)
        println("相似性指标计算结束")
        println("开始计算AUC指标")
        cnauc[i]=CalAUC(train,test,cnsim,n)
        raauc[i]=CalAUC(train,test,rasim,n)
        aaauc[i]=CalAUC(train,test,aasim,n)
        paauc[i]=CalAUC(train,test,pasim,n)
        """
        paauc=CalAUC(train,test,pasim,n)
        lhn1auc=CalAUC(train,test,lhn1sim,n)
        jaccardauc=CalAUC(train,test,jaccardsim,n)
        lpauc=CalAUC(train,test,lpsim,n)
        """
        #miauc[i]=CalAUC(train,test,misim,n)
        println("AUC指标计算结束")
        println("开始计算aupr指标")
        cnpre[i] = CalPresion(train,test,cnsim,TopL)
        aapre[i] = CalPresion(train,test,aasim,TopL)
        rapre[i] = CalPresion(train,test,rasim,TopL)
        papre[i] = CalPresion(train,test,pasim,TopL)
        """
        papre = CalPresion(train,test,pasim,TopL)
        lhn1pre = CalPresion(train,test,lhn1sim,TopL)
        jaccardpre = CalPresion(train,test,jaccardsim,TopL)
        lppre = CalPresion(train,test,lpsim,TopL)
        """
        #mipre[i] = CalPresion(train,test,misim,TopL)

        df1[:CNauc] = cnauc
        df1[:AAauc] = aaauc
        df1[:RAauc] = raauc
        df1[:PAauc] = paauc
        """
        df1[:PAauc] = paauc
        df1[:LHN1auc] = lhn1auc
        df1[:Jaccardauc] = jaccardauc
        df1[:LPauc] = lpauc
        """
        #df1[:MIauc] = miauc
        #println(df1)
        CSV.write(".\\data\\$(datasets[j])100CNAARAPAauc.csv", df1)
        #df2 = DataFrame()
        df2[:CNpre] = cnpre
        df2[:AApre] = aapre
        df2[:RApre] = rapre
        df2[:PApre] = papre
        """
        df2[:PApre] = papre
        df2[:LHN1pre] = lhn1pre
        df2[:Jaccardpre] = jaccardpre
        df2[:LPpre] = lppre
        """
        #df2[:MIpre] = mipre
        #println(df2)
        CSV.write(".\\data\\$(datasets[j])100CNAARAPApre.csv", df2)
        println("aupr计算结束")
    end

    #df[1]表示数据框df中第1列的数据和变量=df[:var1]var1表示列名
end
