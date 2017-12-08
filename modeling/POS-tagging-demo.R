library(NLP)
library(openNLP)
extractPOS <- function(x, thisPOSregex) {
  x <- as.String(x)
  wordAnnotation <- annotate(x, list(Maxent_Sent_Token_Annotator(), Maxent_Word_Token_Annotator()))
  POSAnnotation <- annotate(x, Maxent_POS_Tag_Annotator(), wordAnnotation)
  POSwords <- subset(POSAnnotation, type == "word")
  tags <- sapply(POSwords$features, '[[', "POS")
  thisPOSindex <- grep(thisPOSregex, tags)
  tokenizedAndTagged <- sprintf("%s/%s", x[POSwords][thisPOSindex], tags[thisPOSindex])
  untokenizedAndTagged <- paste(tokenizedAndTagged, collapse = " ")
  untokenizedAndTagged
}

text= data_clean$comb_text[1]
adj=lapply(text, extractPOS, "JJ")
num_adj=sapply(gregexpr("\\JJ", adj), length) # Number of adjectives in a review
num_words = sapply(gregexpr("\\W+", text), length) + 1
perc_adj = num_adj/num_words # 11.3% of the words in this review are adjective
