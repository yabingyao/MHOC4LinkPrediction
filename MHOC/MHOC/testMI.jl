using DataFrames
using CSV

include("read_undir_txt.jl")
#include("similarity_index.jl")
include("DivideNet_notconnected.jl")
include("AUC.jl")
include("Calaupr.jl")
include("CalMI.jl")
#datasets = ["USAir.txt","NS.txt","PB.txt","Yeast.txt","Celegans.txt","FWFB.txt","Power.txt","Router.txt"]
#datasets = ["arenas-email_1133_5451.txt","blogs_1222_16714.txt"]
#datasets = ["Karate_34_78.txt"]
datasets = ["arenas-email_1133_5451.txt"]
#datasets = ["blogs_1222_16714.txt"]
counts = 1
#nsiauc = zeros(Float64,1)
#nsipre = zeros(Float64,1)
homipre = zeros(Float64,counts)
homiauc = zeros(Float64,counts)
TopL = 100
segama = 0.0001
n = 670000

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
        println("开始计算相似性指标")
        println("HigherOrderMI相似性指标计算开始")
        homisim = MI(train)
        #nsisim = NSI(train)
        println("HigherOrderMI相似性指标计算结束")
        println("开始计算AUC指标")
        homiauc[i]=CalAUC(train,test,homisim,n)
        #nsiauc=CalAUC(train,test,nsisim,n)
        println("AUC指标计算结束")
        df1 = DataFrame()
        df2 = DataFrame()
        df1[:MIauc] = homiauc
        #df1[:NSIauc] = nsiauc
        println(df1)
        #CSV.write(".\\data\\$(datasets[j])MIauc.csv", df1)
        println("开始计算aupr指标")
        homipre[i] = CalPresion(train,test,homisim,TopL)
        #nsipre = CalPresion(train,test,nsisim,TopL)
        df2[:MIpre] = homipre
        println(df2)
        #CSV.write(".\\data\\$(datasets[j])MI1pre.csv", df2)
        println("aupr计算结束")
    end
    #df[1]表示数据框df中第1列的数据和变量=df[:var1]var1表示列名
end
