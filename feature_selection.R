library(caret)
library(tidyverse) 
library(readr)

# example for one participant 
p117129 <- read_csv("~/Nextcloud/thesis/p117129_2704.csv")

# manual removal of irrelevant features
p117129 <- subset(p117129, select=c(com.instagram.android_min,
                                  com.whatsapp_min,
                                  APP_USAGE_min,
                                  APPS_OPENED_number,
                                  COMMUNICATION_min,
                                  PHOTOGRAPHY_min,
                                  SOCIAL_min,
                                  com.spotify.music_min,
                                  MUSIC_AUDIO_min,
                                  PRODUCTIVITY_min,
                                  NEWS_MAGAZINES_min,
                                  HEALTH_FITNESS_min,
                                  com.google.android.youtube_min,
                                  VIDEOPLAYERS_EDITORS_min,
                                  Cluster_HOME_min,
                                  TIME_STATIONARY_min,
                                  UNIQUE_STAYPOINTS_number,
                                  Cluster_2_min,
                                  Cluster_1_min,
                                  TOTAL_MACHASHES_number,
                                  UNIQUE_MACHASHES_number,
                                  BLUETOOTH_TOTAL_MACHASHES_number,
                                  BLUETOOTH_UNIQUE_MACHASHES_number,
                                  CALL_TOTAL_min,
                                  CALL_outgoing_min,
                                  MISSED_CALLS_number,
                                  CALL_TOTAL_number,
                                  CALL_outgoing_number,
                                  CALL_UNIQUE_CONTACTS_number,
                                  LIGHT_LUX_mean,
                                  LIGHT_LUX_std,
                                  SCREEN_onLocked_number,
                                  SCREEN_onUnlocked_number,
                                  SMS_received_number,
                                  SMS_UNIQUE_CONTACTS_number,
                                  de.web.mobile.android.mail_min,
                                  com.facebook.katana_min,
                                  com.google.android.apps.maps_min,               
                                  TRAVEL_LOCAL_min,
                                  FINANCE_min,
                                  com.huawei.health_min,
                                  ENTERTAINMENT_min,
                                  Cluster_9_min,
                                  Cluster_8_min,
                                  Cluster_7_min,
                                  Cluster_5_min,
                                  Cluster_4_min,
                                  CALL_incoming_min,
                                  CALL_incoming_number,
                                  Cluster_6_min,
                                  Cluster_3_min,
                                  com.ing.mobile_min,
                                  com.android.email_min,
                                  com.facebook.pages.app_min,
                                  com.Slack_min
                         ))

# removing features with nearly zero variance
nzv <- nearZeroVar(p117129[,11:ncol(p117129)], saveMetrics= TRUE)
rownames(nzv %>% filter(nzv==TRUE))
nzv_idx <- nearZeroVar(p117129[,11:ncol(p117129)])
p117129 <- cbind(p117129[,1:10],
                 p117129[,11:ncol(p117129)][, -nzv_idx])

# manual inspection and removal of features with >0.8 correlation with another feature
corr <- cor(p117129)
p117129_filtered <- subset(p117129,
                           select=-c(SOCIAL_min,
                                     MUSIC_AUDIO_min,
                                     VIDEOPLAYERS_EDITORS_min,
                                     BLUETOOTH_TOTAL_MACHASHES_number,
                                     CALL_TOTAL_min,
                                     CALL_TOTAL_number,
                                     SCREEN_onLocked_number
                                     ))

# dropping features with over 80% of zero entries 
# and features with a mean of less than four minutes across non-zero observations 
percent_0 <- list()
mean_non0 <- list()
for (i in names(p117129_filtered)){
  m <- mean(!p117129_filtered[, i])
  percent_0 <- append(percent_0, m)
  n <- sum(p117129_filtered[, i])/colSums(p117129_filtered != 0)[i]
  mean_non0 <- append(mean_non0, n)
}
a <- cbind(names(p117129_filtered), percent_0,mean_non0)
dropcols <- list()
for (i in names(p117129_filtered)){
  
  if(str_sub(i,start=-3)=="min"&(sum(p117129_filtered[, i],na.rm = TRUE)/colSums(p117129_filtered != 0,na.rm = TRUE)[i])<4){
    dropcols <- append(dropcols, i)
  } else if(mean(!p117129_filtered[, i],na.rm = TRUE)>0.8&(sum(p117129_filtered[, i],na.rm = TRUE)/colSums(p117129_filtered != 0,na.rm = TRUE)[i])<30){
    dropcols <- append(dropcols, i)
    
  }
  
}
p117129_filtered <- p117129_filtered[ , -which(names(p117129_filtered) %in% unlist(dropcols))]

# scaling
passive_scl <- preProcess(p117129_filtered, method = "scale")
p117129_filtered <- predict(passive_scl, p117129_filtered)
p117129_filtered_passive <- cbind(p117129[,5:8],p117129_filtered)
p117129_filtered_passive <- cbind(p117129[,10],p117129_filtered_passive)

write.csv(p117129_filtered_passive,"~/Nextcloud/thesis/p117129_scaled2704.csv", row.names = FALSE)