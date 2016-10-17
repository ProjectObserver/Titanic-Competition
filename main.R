library(randomForest)
library(stats)

# load training data and test date set
ds_tn <- read.csv("train.csv")
#Actual test data
ds_tt <- read.csv("test.csv")
ds_tt <- cbind(factor(rep(0,nrow(ds_tt))),ds_tt)
colnames(ds_tt)[1] = "survived"

## enum for categorical variables......
ds_tt[,"sex"] <- factor(ds_tt[,"sex"])
ds_tt$embarked[ds_tt$embarked == ""]=NA
ds_tt[,"embarked"] <- factor(ds_tt[,"embarked"])

## enum for categorical variables
ds_tn[,"survived"]<-factor(ds_tn[,"survived"])
ds_tn[,"sex"] <- factor(ds_tn[,"sex"])
ds_tn$embarked[ds_tn$embarked == ""]=NA
ds_tn[,"embarked"] <- factor(ds_tn[,"embarked"])

## suppress some of the irrelevant variables
###"ticket","cabin","name"
ds_tn <- ds_tn[,c(-3,-8,-10)]

# only the male one
#ds_tn <- subset(ds_tn, ds_tn$age < 10)

#Actual test data
ds_tt <- ds_tt[,c(-3,-8, -10)]
#ds_tn <- rfImpute(survived ~ ., ds_tn)

# split data into training set and test set
#ds_s <- split(ds_tn, sample(1:2, length(ds_tn), replace = T, prob=c(1,2)))

#ds_tn_s <- data.frame(ds_s[2])
#colnames(ds_tn_s) = c("survived","pclass","sex","age","sibsp","parch","fare","embarked")
#ds_tt <- data.frame(ds_s[1])
#colnames(ds_tt) = c("survived","pclass","sex","age","sibsp","parch","fare","embarked")

#Actual test data
ds_tn_s <- ds_tn

## rough fix NA
ds_tn_s <- na.roughfix(ds_tn_s)
ds_tt <- na.roughfix(ds_tt)

# no pca
ds_tn_s2 <- ds_tn_s
ds_tt_2 <- ds_tt

# adding more feature
ds_tn_s2$minor = ds_tn_s2$age < 10
ds_tt_2$minor = ds_tt_2$age < 10

#ds_tn_s2$old = ds_tn_s2$age > 60
#ds_tt_2$old = ds_tt_2$age < 60

#ds_tn_s2$mmale = ((ds_tn_s2$age > 25) && (ds_tn_s2$sex == "male"))
#ds_tt_2$mmale = ((ds_tn_s2$age > 25) && (ds_tn_s2$sex == "male"))

# feature scaling
ds_tn_s2$age = (ds_tn_s2$age - mean(ds_tn_s2$age))/(max(ds_tn_s2$age)-min(ds_tn_s2$age))+0.5 
ds_tt_2$age = (ds_tt_2$age - mean(ds_tt_2$age))/(max(ds_tt_2$age)-min(ds_tt_2$age))+0.5 

ds_tn_s2$fare = log(ds_tn_s2$fare + 0.5)
ds_tt_2$fare = log(ds_tt_2$fare + 0.5)
ds_tn_s2$fare = (ds_tn_s2$fare - mean(ds_tn_s2$fare))/(max(ds_tn_s2$fare)-min(ds_tn_s2$fare))+0.5 
ds_tt_2$fare = (ds_tt_2$fare - mean(ds_tt_2$fare))/(max(ds_tt_2$fare)-min(ds_tt_2$fare))+0.5 

ds_tn_s2$pclass = (ds_tn_s2$pclass - mean(ds_tn_s2$pclass))/(max(ds_tn_s2$pclass)-min(ds_tn_s2$pclass))+0.5 
ds_tt_2$pclass = (ds_tt_2$pclass - mean(ds_tt_2$pclass))/(max(ds_tt_2$pclass)-min(ds_tt_2$pclass))+0.5 

ds_tn_s2$parch = log(ds_tn_s2$parch + 0.5)
ds_tt_2$parch = log(ds_tt_2$parch + 0.5)
ds_tn_s2$parch = (ds_tn_s2$parch - mean(ds_tn_s2$parch))/(max(ds_tn_s2$parch)-min(ds_tn_s2$parch))+0.5 
ds_tt_2$parch = (ds_tt_2$parch - mean(ds_tt_2$parch))/(max(ds_tt_2$parch)-min(ds_tt_2$parch))+0.5 


# building the tree forest
ds_tn_s2.pred <- ds_tn_s2[,c(-1)]
ds_tn_s2.o <- ds_tn_s2[,1]
ds_tn_s2.rf <- randomForest(ds_tn_s2.pred, ds_tn_s2.o, prox=TRUE, ntree = 500, important=T, corr.bias=T)


## plot the hits and misses for the training set
plot(as.numeric(ds_tn_s2[,"survived"]) + runif(nrow(ds_tn_s2),0,0.3), ds_tn_s2[,"age"], pch=21, xlab="survived", ylab="age",
     bg=c("red", "blue")[as.numeric(factor(ds_tn_s2.rf$predicted))],
     main="Training Hits and Misses")


# predict test set using model
ds_tt_2.pred <- predict(object=ds_tn_s2.rf, newdata=ds_tt_2)

## plot the hits and misses for the test set
plot(as.numeric(ds_tt_2[,"survived"]) + runif(nrow(ds_tt_2),0,0.3), ds_tt_2[,"age"], pch=21, xlab="survived", ylab="age",
     bg=c("red", "blue")[as.numeric(factor(ds_tt_2.pred))],
     main="Test Hits and Misses")

ds_tn_s2.acc = sum(ds_tn_s2[,"survived"]== ds_tn_s2.rf$predicted)/length(ds_tn_s2.rf$predicted)
ds_tt_2.acc = sum(ds_tt_2[,"survived"]== ds_tt_2.pred)/length(ds_tt_2.pred)

print (ds_tn_s2.acc)
print (ds_tt_2.acc)

#output data
write.table(as.matrix(ds_tt_2.pred), file = 'output.csv', sep=",", row.names=F, col.names=F)
