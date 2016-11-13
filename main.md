# Titantic Competiton
Observer  
October 17, 2016  



## Included Package

```r
library(caret)
library(doMC)
library(randomForest)
library(data.table)
library(DMwR)
library(stringr)
library(foreach)
```

## Set Core

```r
# using multiple processing... comment this if it does not work
registerDoMC(cores = 7)
```

## Load Data

```r
dtest<-read.csv("./data/test.csv", header = T)
ds<-read.csv("./data/train.csv", header = T)
dtst = dtest
dtst$Survived=0
dtst$src = 1
ds$src = 0
dall<-rbind(ds,dtst)
dall$Survived = as.factor(dall$Survived)
```

## EDA

```r
# Normalized Fare by Pclass
ds$nFare[ds$Pclass==1] = log(ds$Fare[ds$Pclass==1] +0.01) - log(mean(ds$Fare[ds$Pclass==1])+0.01)
ds$nFare[ds$Pclass==2] = log(ds$Fare[ds$Pclass==2] +0.01) - log(mean(ds$Fare[ds$Pclass==2])+0.01)
ds$nFare[ds$Pclass==3] = log(ds$Fare[ds$Pclass==3] +0.01) - log(mean(ds$Fare[ds$Pclass==3])+0.01)
ds$nFare[is.na(ds$nFare)] = 0

ds_m <- ds[!is.na(ds$Age),]
ggplot(ds_m, aes(SibSp,Age)) + stat_smooth() + geom_point(alpha=0.1)
```

![](main_files/figure-html/EDA-1.png)<!-- -->

```r
ggplot(ds_m, aes(factor(Pclass),Age)) + geom_boxplot()
```

![](main_files/figure-html/EDA-2.png)<!-- -->

```r
ggplot(ds_m, aes(Parch,Age)) + stat_smooth() + geom_point(alpha=0.1)
```

![](main_files/figure-html/EDA-3.png)<!-- -->

```r
ggplot(ds_m[!(ds_m$Embarked==""),], aes(x=Age, fill=Embarked, colour=Embarked)) + geom_density(adjust=1, alpha=0.1)
```

![](main_files/figure-html/EDA-4.png)<!-- -->

```r
ggplot(ds_m[ds_m$Pclass==1,], aes(Fare,Age)) + stat_smooth() + geom_point(alpha=0.1)
```

![](main_files/figure-html/EDA-5.png)<!-- -->

```r
ggplot(ds_m[ds_m$Pclass==2,], aes(Fare,Age)) + stat_smooth() + geom_point(alpha=0.1)
```

![](main_files/figure-html/EDA-6.png)<!-- -->

```r
ggplot(ds_m[ds_m$Pclass==3,], aes(Fare,Age)) + stat_smooth() + geom_point(alpha=0.1)
```

![](main_files/figure-html/EDA-7.png)<!-- -->

```r
ggplot(ds_m, aes(Embarked, Survived)) +geom_jitter(alpha=0.1) + geom_point(alpha=0.1)
```

![](main_files/figure-html/EDA-8.png)<!-- -->

```r
ggplot(ds_m, aes(SibSp, Survived)) +geom_jitter(alpha=0.1) + geom_point(alpha=0.1)
```

![](main_files/figure-html/EDA-9.png)<!-- -->

```r
ggplot(ds_m, aes(Pclass, Survived)) +geom_jitter(alpha=0.1) + geom_point(alpha=0.1)
```

![](main_files/figure-html/EDA-10.png)<!-- -->

```r
ggplot(ds_m, aes(Parch, Survived)) +geom_jitter(alpha=0.1) + geom_point(alpha=0.1)
```

![](main_files/figure-html/EDA-11.png)<!-- -->

```r
ds_m$AgeDecile = cut(ds_m$Age, c(-2000,7,59,200))

ggplot(ds_m[ds_m$Pclass==1,], aes(x=(nFare), fill=AgeDecile, colour=AgeDecile)) + geom_density(adjust=1, alpha=0.1)
```

![](main_files/figure-html/EDA-12.png)<!-- -->

```r
ggplot(ds_m[ds_m$Pclass==2,], aes(x=(nFare), fill=AgeDecile, colour=AgeDecile)) + geom_density(adjust=1, alpha=0.1)
```

![](main_files/figure-html/EDA-13.png)<!-- -->

```r
ggplot(ds_m[ds_m$Pclass==3 & as.integer(ds_m$AgeDecile)<3,], aes(x=(nFare), fill=AgeDecile, colour=AgeDecile)) + geom_density(adjust=1, alpha=0.1)
```

![](main_files/figure-html/EDA-14.png)<!-- -->

## Supporting Functions

```r
#### function to count number of space in the string
getCountSp <- function(str){
  m<-gregexpr("\\s", str, perl=T)
  return(unlist(length(regmatches(str,m)[[1]])))
}

#### function to generate lookup for cnt per ticket
generate_cnt_by_Ticket<-function(dtf){
  # Create variable for prob to Survive within the same cabin: same cabin usually has same outcome
  dtf<-data.frame(Ticket=dtf$Ticket, Survived=as.integer(dtf$Survived) -1)
  dt<-data.table(dtf)
  lookup<-dt[,list(cnt_ticket = .N), by=Ticket]
  return(lookup)
}

#### function to generate samples from Beta distribution; 
###### piror dist. = Beta(2,4) : p_Survived = 1/3; posterior = Beta(1+Survived, 1+dead)
###### the prior probability is used to control how much information of p_Survived. 
###### p_Survived is a average survived per ticket of known cases. Although p_Survived is a leakage of data, 
###### the factor captured the same ticket number tends to have the similar outcome.
###### Using the piror, the influence of this variable could be reduced. 
###### This variable should have similar predictive power as title: Mr
generate_BAY_p_ds<-function(i, dtest){
  p <- rbeta(1, dtest$SurvivedTotal[i] + 2, dtest$nsample[i] - dtest$SurvivedTotal[i] + 4)
  return (p)
}

#### function to generate lookup for cnt per GrpT and number of survived per GrpT
generate_n_Survived_GrpT<-function(dtf){
  # Create variable for prob to Survive within the same cabin: same cabin usually has same outcome
  df<-data.frame(GrpT=dtf$GrpT, Survived=as.integer(dtf$Survived) -1, isMr = dtf$isMr)
  dt<-data.table(df)
  lookup<-dt[,list(SurvivedTotal = mean(Survived) * .N, nsample = .N), by=list(GrpT, isMr)]
  return(lookup)
}

#### function to add information in lookup by name into data set
add_info_by_name <- function(dtr, name_, lookup){
  dt <- merge(x=dtr, y=lookup, by=name_, all.x=T)
  return(dt)
}

#### function to extract cabin number
get_CabinNum<-function(ll){
  v <- as.numeric(unlist(strsplit(unlist(strsplit(ll, " "))[length(unlist(strsplit(ll, " ")))], "[A-Z]"))[2])
  return (v)
}

#### function to create variables before the Age prediction
pretreatment <- function(ds){
  ds$Pclass = as.factor(ds$Pclass)
  
  # Get the title as predictor for Age
  c<-str_locate(ds$Name," ([a-z;A-Z]*)\\.")
  ds$title=substr(ds$Name, c[,1]+1, c[,2]-1)
  ds$title[ds$title=="Ms"] = "Miss"
  ds$title[!(ds$title=="Mr" | ds$title=="Miss" | ds$title=="Mrs" | ds$title=="Master")] = "Other"
  ds$title = as.factor(ds$title)
  
  c<-str_locate(ds$Name,"^([a-z;A-Z;']*)[, -]")
  ds$lastname=as.factor(substr(ds$Name, c[,1], c[,2]-1))
  
  # Create dummy variable isSingle... 
  ds$isSingle <- as.factor(ds$Parch==0)
  ds$CabinLet = substr(ds$Cabin, 1, 1)
  ds$CabinLet[ds$CabinLet=="T"] = ""
  ds$CabinLet = as.factor(ds$CabinLet)
  ds$CabinCnt = as.integer(!ds$Cabin=="") + sapply(ds$Cabin,getCountSp, simplify=T)
  ds$CabinNum = -10
  ll <- as.character(ds$Cabin[!as.character(ds$Cabin)==""])
  ds$CabinNum[!as.character(ds$Cabin)==""] = CabinNum<-sapply(ll, get_CabinNum)
  ds$CabinNum[is.na(ds$CabinNum)] = -10
  ds$familySz = ds$SibSp + ds$Parch
  
  ds$nFare[ds$Pclass==1] = log(ds$Fare[ds$Pclass==1] +0.01) - log(mean(ds$Fare[ds$Pclass==1])+0.01)
  ds$nFare[ds$Pclass==2] = log(ds$Fare[ds$Pclass==2] +0.01) - log(mean(ds$Fare[ds$Pclass==2])+0.01)
  ds$nFare[ds$Pclass==3] = log(ds$Fare[ds$Pclass==3] +0.01) - log(mean(ds$Fare[ds$Pclass==3])+0.01)
  ds$nFare[is.na(ds$nFare)] = 0
  
  ds$r_SibSp_Parch = ((ds$SibSp+0.01)/(ds$SibSp+ds$Parch+0.01))
  
  # Similar location would have similar outcome
  # Get total number of people per ticket
  lookup<-generate_cnt_by_Ticket(ds)
  ds<-add_info_by_name(ds,"Ticket",lookup)
  # Define the group ticket that has more than 2 people
  ds$GrpT = "XXXX"
  ds$GrpT[ds$cnt_ticket>1] = as.character(ds$Ticket[ds$cnt_ticket>1])
  
  # For the ungrouped passenger, group them by cabin
  df<-table(ds$Cabin[as.character(ds$GrpT)=='XXXX'])
  lu<-as.data.frame(names(df[df>1]))
  names(lu)<-c("Cabin")
  lu$flag <- 1
  lu<-lu[!lu$Cabin=="",]
  ds<-merge(x=ds, y=lu, by="Cabin", all.x=T)
  ds$GrpT[!is.na(ds$flag) & as.character(ds$GrpT)=='XXXX']<-as.character(ds$Cabin[!is.na(ds$flag) & as.character(ds$GrpT)=='XXXX'])

  # For the rest of ungrouped passenger, group them by last name
  df<-table(ds$lastname[as.character(ds$GrpT)=='XXXX'])
  lu<-as.data.frame(names(df[df>1]))
  names(lu)<-c("lastname")
  lu$flag1 <- 1
  lu<-lu[!lu$lastname=="",]
  ds<-merge(x=ds, y=lu, by="lastname", all.x=T)
  ds$GrpT[!is.na(ds$flag1) & as.character(ds$GrpT)=='XXXX']<-as.character(ds$lastname[!is.na(ds$flag1) & as.character(ds$GrpT)=='XXXX'])

  return(ds)
}

#### Try to predict Fare for ticket with single passenger only... 
##### Want to use that to deduce fare for individaul in combined ticket... Not sucessful
md_Fare <- function(de) {
  trf_Fare<- lm(I(log(Fare+0.001))~ title + CabinLet * I(as.factor(CabinNum)) + CabinCnt + Embarked , data=de)

  print(summary(trf_Fare))
  return(trf_Fare)
}

#### function to do Age Prediction
  # Predict Age for missings
  # if isSingle and title is Miss, could be older
  # if isSingle and has a lot of SibSp, could be younger
  # if Pclass is controlled, Fare should reflect the age
  # using log(Age) because Age distribution is right-skewed... except the baby... but mostly right skewed
md_Age <- function(ds_m) {
  ds <- rbind(ds_m, ds_m[ds_m$Age > 60,])
  trf_Age<- randomForest(
                I(log(Age+0.0001)) ~ title + isSingle*SibSp + I(Parch<=2):I(SibSp>1) + Parch:familySz + 
                             SibSp + Pclass + r_SibSp_Parch + CabinLet + CabinNum + CabinCnt + 
                             Embarked + cnt_ticket + nFare,
                data=ds, importance=TRUE,proximity=TRUE,ntree=701
        )
  print(varImp(trf_Age))
  return(trf_Age)
}

#### function to create variables after the Age prediction
psttreatment <- function(trf_Age, ds) {
  ds_m <- ds
  ds_m$AgePred <- exp(predict(trf_Age, ds_m))
  ds_m$AgeDecile = cut(ds_m$Age, c(-2000,11,15,18,30,49,59,200))
  ds_m$AgePredDecile = cut(ds_m$AgePred, c(-2000,11,15,18,30,49,59,200))
  print(table(ds_m$AgeDecile[!is.na(ds_m$Age)],ds_m$AgePredDecile[!is.na(ds_m$Age)]))
  ggplot(ds_m, aes(AgePred, Age)) + geom_point(alpha=0.1) + geom_smooth(method='lm',formula=y~x)
  # replace NA
  ds_m$noAge <- 0
  ds_m$noAge[is.na(ds_m$Age)]<-1
  ds_m$Age[is.na(ds_m$Age)]<-ds_m$AgePred[is.na(ds_m$Age)]
  # cut by decile
  ds_m$AgeDecile = cut(ds_m$Age, c(-2000,11,15,18,30,49,59,200))
  ds_m$isMr = (ds_m$title == "Mr" & ds_m$Age > 17)
  return(ds_m)
}

#### function to run the prediction
run_prediction <- function(tgbm2, trf, tsvm, dtest){
  ## Run the data through the model
  dtest$pred.tgbm2 = predict(tgbm2, dtest, "raw")
  dtest$pred.rf = predict(trf, dtest, "raw")
  dtest$pred.svm = predict(tsvm, dtest, "raw")
  return(dtest)
}
```

## Data Manipulation

```r
dall<-pretreatment(dall)
dall$GrpT = as.factor(dall$GrpT)
trf_Age <- md_Age(dall[!is.na(dall$Age),])
```

```
##                 Overall
## title         66.924704
## isSingle      13.691352
## SibSp         14.356139
## Pclass        32.968882
## r_SibSp_Parch 16.046554
## CabinLet      16.137229
## CabinNum      18.700282
## CabinCnt       7.404100
## Embarked      19.242321
## cnt_ticket    35.486976
## nFare         30.331919
## I(Parch <= 2)  6.455269
## I(SibSp > 1)  10.179720
## Parch         16.293392
## familySz      15.216956
```

```r
dall<-psttreatment(trf_Age,dall)
```

```
##              
##               (-2e+03,11] (11,15] (15,18] (18,30] (30,49] (49,59] (59,200]
##   (-2e+03,11]          88       0       1       1       0       0        0
##   (11,15]               6       5       7       7       0       0        0
##   (15,18]               3       3       5      63       4       0        0
##   (18,30]               2       2       5     300     107       0        0
##   (30,49]               0       0       1     106     216       4        0
##   (49,59]               0       0       0       9      52       9        0
##   (59,200]              0       0       0       3      14      17        6
```

```r
#trf_Fare <- md_Fare(de <- dall[dall$cnt_ticket==1 & !is.na(dall$Fare),])

ds <- dall[dall$src == 0,]
dtest <- dall[dall$src == 1,]

ggplot(ds, aes(factor(title),Age)) + geom_boxplot()
```

![](main_files/figure-html/ManupulateData-1.png)<!-- -->

```r
ggplot(ds, aes(factor(title),Survived))  + geom_point(alpha=0.1) +geom_jitter(alpha=0.1)
```

![](main_files/figure-html/ManupulateData-2.png)<!-- -->

```r
ggplot(ds, aes(CabinLet, Survived)) +geom_jitter(alpha=0.1) + geom_point(alpha=0.1)
```

![](main_files/figure-html/ManupulateData-3.png)<!-- -->

## Modeling

```r
ctrl = trainControl(method="repeatedcv", number=20, repeats=10, selectionFunction = "oneSE")
in_train = createDataPartition(ds$Survived, p=1, list=FALSE)

#dtr <- ds[in_train,]
dtr<-ds

lookup<-generate_n_Survived_GrpT(dtr)
dtr<-add_info_by_name(dtr,c('GrpT','isMr'), lookup)
dtr$nsample[dtr$GrpT == "XXXX"] = dtr$nsample[dtr$GrpT == "XXXX"] / 600
dtr$SurvivedTotal[dtr$GrpT == "XXXX"] = dtr$SurvivedTotal[dtr$GrpT == "XXXX"] / 600
dtr$p_Survived<-sapply(1 : length(dtr$nsample), generate_BAY_p_ds, dtest =dtr)
ggplot(dtr, aes(x=p_Survived, fill=Survived, colour=Survived)) + geom_histogram(alpha=0.5)
```

![](main_files/figure-html/Modeling-1.png)<!-- -->

```r
ggplot(dtr, aes(x=p_Survived, fill=Survived, colour=Survived)) + geom_density(adjust=1, alpha=0.1)
```

![](main_files/figure-html/Modeling-2.png)<!-- -->

```r
# train model using multiple methods
tune_grid <-  expand.grid(interaction.depth = c(1, 3, 9, 11),
                          n.trees = (1:30)*10,
                          shrinkage = 0.1,
                          n.minobsinnode = 10)
tgbm2 = 
  train(
    Survived ~  Sex + p_Survived + cnt_ticket + title + Pclass + CabinLet + CabinNum + CabinCnt + 
                noAge + AgeDecile + SibSp + Parch + Embarked + familySz + nFare + I(nFare/cnt_ticket) + 
                r_SibSp_Parch, 
    data=dtr, method="gbm", tuneGrid=tune_grid, preProc = c("center", "scale"), metric="Kappa", 
    trControl=ctrl, verbose=FALSE
  )
varImp(tgbm2)
```

```
## gbm variable importance
## 
##   only 20 most important variables shown (out of 34)
## 
##                      Overall
## Sexmale             100.0000
## titleMr              61.2048
## p_Survived           60.7182
## Pclass3              30.4270
## CabinNum             24.4326
## cnt_ticket           17.7509
## nFare                11.4690
## familySz              9.3618
## EmbarkedS             6.5665
## r_SibSp_Parch         4.2928
## I(nFare/cnt_ticket)   4.2540
## titleOther            4.1932
## AgeDecile(18,30]      2.3851
## AgeDecile(30,49]      1.4557
## EmbarkedQ             1.2681
## CabinLetC             1.1230
## EmbarkedC             1.0743
## CabinLetE             1.0427
## titleMrs              1.0322
## noAge                 0.6713
```

```r
x <- with (dtr,
  cbind(
        Sex, p_Survived, cnt_ticket, title, Pclass, CabinLet, CabinNum, CabinCnt, noAge, AgeDecile, 
        SibSp, Parch, Embarked, familySz, nFare, r_SibSp_Parch
  )
)

bestmtry <- tuneRF(x, dtr$Survived, stepFactor=1.5, improve=1e-4, ntree=1001, doBest = F)
```

```
## mtry = 4  OOB error = 17.28% 
## Searching left ...
## mtry = 3 	OOB error = 16.61% 
## 0.03896104 1e-04 
## mtry = 2 	OOB error = 17.28% 
## -0.04054054 1e-04 
## Searching right ...
## mtry = 6 	OOB error = 16.61% 
## 0 1e-04
```

![](main_files/figure-html/Modeling-3.png)<!-- -->

```r
tune_grid <- expand.grid(.mtry=c(bestmtry[bestmtry[,2] == min(bestmtry[,2]),1]))

trf = 
  train(
    Survived ~  Sex + p_Survived + cnt_ticket + title + Pclass + CabinLet+ CabinNum + CabinCnt + 
                noAge + AgeDecile + SibSp + Parch + Embarked + familySz + nFare + I(nFare/cnt_ticket) + 
                r_SibSp_Parch, 
    data=dtr, method="rf", metric="Kappa", tuneGrid=tune_grid, trControl=ctrl, preProc=c("center", "scale"), 
    verbose=FALSE, ntree=1001
  )

#method="svmLinear","svmPoly" svmRadial
tsvm = 
  train(
    Survived ~  Sex + p_Survived + cnt_ticket + title + Pclass + CabinLet + CabinNum + CabinCnt + 
                noAge + AgeDecile + SibSp + Parch + Embarked + familySz + nFare + I(nFare/cnt_ticket) + 
                r_SibSp_Parch, 
    tuneGrid = data.frame(.C = seq(0,0.95,0.05) + 0.05), data=dtr, method="svmLinear", metric="Kappa", 
    trControl=ctrl, verbose=FALSE, preProc = c("center", "scale")
  )

varImp(trf)
```

```
## rf variable importance
## 
##   only 20 most important variables shown (out of 34)
## 
##                     Overall
## titleMr             100.000
## Sexmale              94.286
## p_Survived           88.616
## CabinNum             35.237
## titleMiss            33.810
## cnt_ticket           32.785
## I(nFare/cnt_ticket)  32.777
## nFare                32.592
## titleMrs             29.234
## Pclass3              25.761
## CabinCnt             22.872
## familySz             22.245
## r_SibSp_Parch        16.760
## SibSp                14.003
## Parch                10.590
## EmbarkedS             8.848
## AgeDecile(30,49]      7.790
## EmbarkedC             7.327
## AgeDecile(18,30]      6.622
## Pclass2               6.487
```

## Resampling

```r
resampls = resamples(list(RF = trf,
                          GBM = tgbm2, SVM = tsvm
                         ))
difValues = diff(resampls)
summary(difValues)
```

```
## 
## Call:
## summary.diff.resamples(object = difValues)
## 
## p-value adjustment: bonferroni 
## Upper diagonal: estimates of the difference
## Lower diagonal: p-value for H0: difference = 0
## 
## Accuracy 
##     RF GBM       SVM      
## RF     -0.002220  0.002055
## GBM 1             0.004275
## SVM 1  1                  
## 
## Kappa 
##     RF GBM        SVM       
## RF     -0.0056698 -0.0007643
## GBM 1              0.0049055
## SVM 1  1
```

```r
bwplot(resampls, layout=c(2,1))
```

![](main_files/figure-html/Resampling-1.png)<!-- -->

## Generate Summit File for Competition

```r
# Regenerate lookup using all training info
lookup<-generate_n_Survived_GrpT(ds)

# Create variable for prob to Survive with the same ticket
dtest<-add_info_by_name(dtest,c('GrpT','isMr'), lookup)
dtest$SurvivedTotal[is.na(dtest$SurvivedTotal)] = 0
dtest$nsample[is.na(dtest$nsample)] = 0

## Run the data through the model
dtest$nsample[dtest$GrpT == "XXXX"] = dtest$nsample[dtest$GrpT == "XXXX"] / 600
dtest$SurvivedTotal[dtest$GrpT == "XXXX"] = dtest$SurvivedTotal[dtest$GrpT == "XXXX"] / 600

n <- 10001
a <-  foreach (i = 1:n, .combine = rbind) %dopar%
{
  dtest$p_Survived<-sapply(1 : length(dtest$nsample), generate_BAY_p_ds, dtest =dtest)
  dtest<-run_prediction(tgbm2, trf, tsvm, dtest)
  as.integer(dtest$pred.tgbm2) - 1 + as.integer(dtest$pred.rf) - 1 + as.integer(dtest$pred.svm) - 1
}
pred.vote = colSums(a)
hist(pred.vote)
```

![](main_files/figure-html/Generate_Summit-1.png)<!-- -->

```r
dtest$pred.vote = as.integer(pred.vote>=as.integer(n*3/2))

## Format the summit file and save to ./data/pred.csv
dSummit<-as.data.frame(cbind(dtest$PassengerId,as.integer(dtest$pred.vote)))
names(dSummit)=c("PassengerId","Survived")
write.table(dSummit,"./data/pred.csv", row.names=F, col.names = T, sep=",", quote=FALSE)
```
