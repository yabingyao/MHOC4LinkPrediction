using DataFrames
using CSV

include("read_undir_txt.jl")
include("similarity_index.jl")
include("DivideNet_notconnected.jl")
include("AUC.jl")
include("Calaupr.jl")
include("HigherOrderMI.jl")
include("CalMI.jl")
include("CalCNDP.py")
#datasets = ["Karate_34_78.txt","USAir_332_2126.txt","NetScience_379_914.txt","celegans_297_2148.txt"]
#datasets = ["Karate_34_78.txt","USAir_332_2126.txt","NetScience_379_914.txt","celegans_297_2148.txt","arenas-email_1133_5451.txt","blogs_1222_16714.txt"]
datasets = ["arenas-email_1133_5451.txt","blogs_1222_16714.txt"]
#datasets = ["Karate_34_78.txt"]
#datasets = ["arenas-email_1133_5451.txt"]
#datasets = ["blogs_1222_16714.txt"]
counts = 100
TopL = 100
segama = 0.0001
n = 670000
cnauc = zeros(Float64,counts)
cnpre = zeros(Float64,counts)
homipre = zeros(Float64,counts)
homiauc = zeros(Float64,counts)
aaauc = zeros(Float64,counts)
aapre = zeros(Float64,counts)
raauc = zeros(Float64,counts)
rapre = zeros(Float64,counts)
miauc = zeros(Float64,counts)
mipre = zeros(Float64,counts)
paauc = zeros(Float64,counts)
papre = zeros(Float64,counts)
df1 = DataFrame()
df2 = DataFrame()
for j = 1 : length(datasets)
    println("正在加载数据集",datasets[j])
    data = load_example_data(datasets[j])
    println("数据集加载完成")
    for i = 1 : counts
        println("第",i,"次")
        #println("开始划分训练集和测试集")
        net = DivideNetNoCon(data,0.9)
        #println("训练集划分结束")
        train = net[1]
        test = net[2]
        #println("开始计算相似性指标")
        println("HigherOrderMI相似性指标计算开始")
        homisim = CalHigherOrderMI(train)
        println("HigherOrderMI相似性指标计算结束")
        println("MI相似性指标计算开始")
        misim = MI(train)
        println("MI相似性指标计算结束")
        #println("CNAARAPA相似性指标计算开始")
        cnsim = CN(train)
        #println("cn相似性指标计算结束")
        aasim = AA(train)
        #println("aa相似性指标计算结束")
        rasim = RA(train)
        #println("ra相似性指标计算结束")
        pasim = PA(train)
        #println("pa相似性指标计算结束")
        pasim = PA(train)
        #println("pa相似性指标计算结束")
        println("开始计算AUC指标")
        homiauc[i]=CalAUC(train,test,homisim,n)
        miauc[i]=CalAUC(train,test,misim,n)
        cnauc[i]=CalAUC(train,test,cnsim,n)
        raauc[i]=CalAUC(train,test,rasim,n)
        aaauc[i]=CalAUC(train,test,aasim,n)
        paauc[i]=CalAUC(train,test,pasim,n)
        #println("AUC指标计算结束")
        df1.CNauc = cnauc
        df1.AAauc = aaauc
        df1.RAauc = raauc
        df1.PAau = paauc
        df1.HighOrderMIauc = homiauc
        df1.MIauc = miauc
        CSV.write("~/linkprediction/data/$(datasets[j])auc.csv", df1)
        println("开始计算AUC-PRC指标")
        homipre[i] = CalPresion(train,test,homisim,TopL)
        mipre[i] = CalPresion(train,test,misim,TopL)
        cnpre[i] = CalPresion(train,test,cnsim,TopL)
        aapre[i] = CalPresion(train,test,aasim,TopL)
        rapre[i] = CalPresion(train,test,rasim,TopL)
        papre[i] = CalPresion(train,test,pasim,TopL)
        df2.CNpre = cnpre
        df2.AApre = aapre
        df2.RApre = rapre
        df2.PApre = papre
        df2.HighOrderMIpre = homipre
        df2.MIpre = mipre
        CSV.write("~/linkprediction/data/$(datasets[j])pre.csv", df2)
        println("aupr计算结束")
    end
    println(df1)
    println(df2)
    #df[1]表示数据框df中第1列的数据和变量=df[:var1]var1表示列名
end
