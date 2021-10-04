
Data <- read.csv('/Users/toneill/N159/trimmed_ksoll_training.csv')

library(ISLR2)
library(tidyverse)

# set which are pms with over 90% prob
if_pms <- Data$pms_membership_prob >= 0.9
Data$pms <- if_pms*1

##make R view hilow as categorical and check dummy coding
Data$pms<-factor(Data$pms)
#is.factor(Data$pms)
#contrasts(Data$pms)

##MVN tests 
# both fail assumptions of normality
# but okay since mostly concerned with classification here
ICS::mvnorm.kur.test(Data[1:5])
ICS::mvnorm.skew.test(Data[1:5])


##create boxplots
ggplot(Data, aes(x=pms, y=m_f555w,group=pms))+
  geom_boxplot()
##
ggplot(Data, aes(x=pms, y=m_f775w,group=pms))+
  geom_boxplot()
##
ggplot(Data, aes(x=pms, y=m_f110w,group=pms))+
  geom_boxplot()
##
ggplot(Data, aes(x=pms, y=m_f160w,group=pms))+
  geom_boxplot()
##
ggplot(Data, aes(x=pms, y=A_v,group=pms))+
  geom_boxplot()

## make pairs plot
pairs(Data[,1:5], col = c(1,2)[Data$pms], lower.panel=NULL)
# very little separation between groups, all filters are linearly correlated
# with each other, no correlation between A_v and filters

##create training and test data
sample.data<-sample.int(nrow(Data), floor(.50*nrow(Data)), replace = F)
train<-Data[sample.data, ]
test<-Data[-sample.data, ]

##Carry out LDA on training data
lda.pms <- MASS::lda(pms ~ ., data=train[c(1:5,7)])
##obtain ouput from LDA
lda.pms
# coeffs: f555 strong positive, 775 strong negative, 
# 110 and 160 weak negative, Av almost = 0

##predictions on training data. 
lda.train <- predict(lda.pms)
##Confusion matrix on training data. Rows represent actual value, cols represent pred value
table(train$pms, lda.train$class)
##accuracy on training data
mean(train$pms == lda.train$class)
##predictions on test data. 
lda.test <- predict(lda.pms,test)
##confusion matrix. By default, threshold is 0.5
table(test$pms,lda.test$class)
##accuracy on test data
mean(test$pms == lda.test$class)


### slightly more accurate for training data (90.7% vs 91.1%)

library(ROCR)
preds<-lda.test$posterior[,2]
rates<-ROCR::prediction(preds, test$pms)
roc_result<-ROCR::performance(rates,measure="tpr", x.measure="fpr")
plot(roc_result, main="ROC Curve for CHD")
lines(x = c(0,1), y = c(0,1), col="red")

# roc curve tells us better than random guessing
auc<-ROCR::performance(rates, measure = "auc")
auc@y.values

# AUC of 0.95415 - wow!!!

#######################################
## trying logistic regression##
#######################################

result_train<-glm(pms~m_f555w+m_f775w+m_f110w+m_f160w+A_v, family=binomial, data=train)
summary(result_train)
# Av fails z test (p=0.7), all others significant

### model without A_v
result_noAv<-glm(pms~m_f555w+m_f775w+m_f110w+m_f160w, family=binomial, data=train)
summary(result_noAv)

deltaG2 <- result_noAv$dev - result_train$dev
1-pchisq(deltaG2,2)
# p value of 0.93 - fail to reject null hypothesis, 
# reduced model (without Av) is superior, can disregard Av

## when go through and try removing one factor at a time
# for all factors, only Av reduced model is an improvement over original -
# for rest, model without Av and with all four filters preferabble

##predicted survival rate for test data based on training data
preds<-predict(result_noAv,newdata=test, type="response")

##produce the numbers associated with classification table
rates<-ROCR::prediction(preds, test$pms)

##store the true positive and false postive rates
roc_result<-ROCR::performance(rates,measure="tpr", x.measure="fpr")

##plot ROC curve and overlay the diagonal line for random guessing
plot(roc_result, main="ROC Curve")
lines(x = c(0,1), y = c(0,1), col="red")

auc<-ROCR::performance(rates, measure = "auc")
auc@y.values

# AUC of 0.9582 - also really good!!!

confusion.mat<-table(test$pms,preds > 0.5)
confusion.mat
# 91.25% accuracy on test data

#######################################################
# test which features most distinguishing for most extincted stars

# top quantile for Av 
top_quant <- quantile(Data$A_v,0.75)
library(dplyr)
Data <- filter(Data, Data$A_v < top_quant)

result_hiav<-glm(pms~m_f555w+m_f775w+m_f110w+m_f160w+A_v, family=binomial, data=Data_HiAv)
summary(result_hiav)

### model without 110 and 160
result_no1116<-glm(pms~m_f555w+m_f775w+A_v, family=binomial, data=Data)
summary(result_no1116)

deltaG2 <- result_no1116$dev - result_hiav$dev
1-pchisq(deltaG2,2)
#fail to reject null hypothesis, 
# reduced model (without 110/160) is superior

####


##create training and test data
sample.data<-sample.int(nrow(Data), floor(.50*nrow(Data)), replace = F)
train<-Data[sample.data, ]
test<-Data[-sample.data, ]

##Carry out LDA on training data
lda.pms <- MASS::lda(pms ~ ., data=train[c(1:5,7)])
##obtain ouput from LDA
lda.pms
# coeffs: f555 strong positive, 775 strong negative, 
# 110 and 160 weak negative, Av almost = 0

##predictions on training data. 
lda.train <- predict(lda.pms)
##Confusion matrix on training data. Rows represent actual value, cols represent pred value
table(train$pms, lda.train$class)
##accuracy on training data
mean(train$pms == lda.train$class)
##predictions on test data. 
lda.test <- predict(lda.pms,test)
##confusion matrix. By default, threshold is 0.5
table(test$pms,lda.test$class)
##accuracy on test data
mean(test$pms == lda.test$class)



preds<-lda.test$posterior[,2]
rates<-ROCR::prediction(preds, test$pms)
roc_result<-ROCR::performance(rates,measure="tpr", x.measure="fpr")
plot(roc_result, main="ROC Curve for CHD")
lines(x = c(0,1), y = c(0,1), col="red")

# roc curve tells us better than random guessing
auc<-ROCR::performance(rates, measure = "auc")
auc@y.values

