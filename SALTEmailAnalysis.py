
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import collections
import os
import random
import numpy as np
from tqdm import tqdm
import sys, email
import pandas as pd
import math
import datetime
import re

#########################################################
# Load Enron dataset
#########################################################

ENRON_EMAIL_DATASET_PATH = './emails.csv'

# load enron dataset
import pandas as pd
emails_df = pd.read_csv(ENRON_EMAIL_DATASET_PATH)
print(emails_df.shape)






#########################################################
# Sort out required email features: date, subject, content
#########################################################

# source https://www.kaggle.com/zichen/explore-enron
## Helper functions
def get_text_from_email(msg):
    '''To get the content from email objects'''
    parts = []
    for part in msg.walk():
        if part.get_content_type() == 'text/plain':
            parts.append( part.get_payload() )
    return ''.join(parts)

import email
# Parse the emails into a list email objects
messages = list(map(email.message_from_string, emails_df['message']))
emails_df.drop('message', axis=1, inplace=True)
# Get fields from parsed email objects
keys = messages[0].keys()
for key in keys:
    emails_df[key] = [doc[key] for doc in messages]
# Parse content from emails
emails_df['Content'] = list(map(get_text_from_email, messages))

# keep only Subject and Content for this exercise
emails_df = emails_df[['Date','Subject','Content']]
#
# #########################################################
# # change wor2vec model to work with Enron emails
# #########################################################
#
#
# point it to our Enron data set
emails_sample_df = emails_df.copy()

import string, re
# clean up subject line
emails_sample_df['Subject'] = emails_sample_df['Subject'].str.lower()
emails_sample_df['Subject'] = emails_sample_df['Subject'].str.replace(r'[^a-z]', ' ')
emails_sample_df['Subject'] = emails_sample_df['Subject'].str.replace(r'\s+', ' ')

# clean up content line
emails_sample_df['Content'] = emails_sample_df['Content'].str.lower()
emails_sample_df['Content'] = emails_sample_df['Content'].str.replace(r'[^a-z]', ' ')
emails_sample_df['Content'] = emails_sample_df['Content'].str.replace(r'\s+', ' ')

# create sentence list
emails_text = (emails_sample_df["Subject"] + ". " + emails_sample_df["Content"]).tolist()

sentences = ' '.join(emails_text)
words = sentences.split()

print('Data size', len(words))


# get unique words and map to glove set
print('Unique word count', len(set(words)))
#
#
# # drop rare words
# vocabulary_size = 5
#
# def build_dataset(words):
#   count = [['UNK', -1]]
#   count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
#   dictionary = dict()
#   for word, _ in count:
#     dictionary[word] = len(dictionary)
#   data = list()
#   unk_count = 0
#   for word in tqdm(words):
#     if word in dictionary:
#       index = dictionary[word]
#     else:
#       index = 0  # dictionary['UNK']
#       unk_count += 1
#     data.append(index)
#   count[0][1] = unk_count
#   reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
#   return data, count, dictionary, reverse_dictionary
#
# data, count, dictionary, reverse_dictionary = build_dataset(words)
#
# del words
# print('Most common words (+UNK)', count[:5])
# print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
#
# ####################################################################
# # find matches with glove
# ####################################################################
GLOVE_DATASET_PATH = './glove.42B.300d.txt'
#
# from tqdm import tqdm
# import string
# embeddings_index = {}
# f = open(GLOVE_DATASET_PATH)
# word_counter = 0
# for line in tqdm(f):
#   values = line.split()
#   word = values[0]
#   if word in dictionary:
#     coefs = np.asarray(values[1:], dtype='float32')
#     embeddings_index[word] = coefs
#   word_counter += 1
# f.close()
#
# print('Found %s word vectors matching enron data set.' % len(embeddings_index))
# print('Total words in GloVe data set: %s' % word_counter)#
#
# #########################################################
# # Check out some clusters
# #########################################################
#
# # create a dataframe using the embedded vectors and attach the key word as row header
# import pandas as pd
# enrond_dataframe = pd.DataFrame(embeddings_index)
# enrond_dataframe = pd.DataFrame.transpose(enrond_dataframe)
#
# # See what it learns and look at clusters to pull out major themes in the data
# CLUSTER_SIZE = 3
# # cluster vector and investigate top groups
# from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters=CLUSTER_SIZE)
# cluster_make = kmeans.fit_predict(enrond_dataframe)
#
# labels = kmeans.predict(enrond_dataframe)
# import collections
# cluster_frequency = collections.Counter(labels)
# print(cluster_frequency)
# cluster_frequency.most_common()
#
# clusters = {}
# n = 0
# for item in labels:
#     if item in clusters:
#         clusters[item].append(list(enrond_dataframe.index)[n])
#     else:
#         clusters[item] = [list(enrond_dataframe.index)[n]]
#     n +=1
#
# for k,v in cluster_frequency.most_common(100):
#   print('\n\n')
#   print('Cluster:', k)
#   print (' '.join(clusters[k]))

  ####################################################
# Load master clusters for all six deparatments
####################################################
#
# LEGAL= ['affirmed','alleged','appeal','appealed','appeals','appellate','attorney','attorneys','bankruptcy','case','cases','charged','charges','civil','claim','claims','complaint','constitutional','constitutionality','constitutionally','copyright','counsel','court','courts','criminal','damages','decision','decree','decrees','defendants','denied','dispute','dissented','dissenting','enforcement','federal','filed','filing','invalidate','invalidated','judge','judgement','judges','judgment','judgments','judicial','judiciary','jurisdiction','jurisprudence','justice','justices','law','laws','lawsuit','lawsuits','lawyer','lawyers','legal','legality','legally','litigation','overrule','overruled','overturn','overturned','overturning','plaintiff','precedent','precedents','prosecutorial','reversed','rights','ruled','ruling','rulings','settlement','settlements','sue','supreme','tribunal','tribunals','unanimous','unconstitutional','upheld','uphold','upholding','upholds','verdict','violation']
#
# COMMUICATIONS=['accessed','ads','alphabetical','alphabetically','archive','archived','archives','below','bookmark','bookmarked','bookmarks','browse','browsing','calendar','catalog','categories','categorized','category','chart','charts','check','classified','classifieds','codes','compare','content','database','details','directories','directory','domain','domains','downloadable','entries','favorites','feeds','free','genealogy','homepage','homepages','hosting','index','indexed','indexes','info','information','keyword','keywords','library','link','linking','links','list','listed','listing','listings','lists','locate','locator','maps','online','page','pages','peruse','portal','profile','profiles','rated','related','resource','results','search','searchable','searched','searches','searching','selections','signup','site','sites','sorted','statistics','stats','subscribing','tagged','testimonials','titles','updated','updates','via','web','webmaster','webpage','webpages','website','websites','wishlist','accountant','careers','clerical','contracting','department','employed','employee','employees','employer','employers','employment','experienced','freelance','fulltime','generalist','hire','hired','hires','hiring','hourly','intern','interviewing','job','jobs','labor','labour','managerial','manpower','office','paralegal','personnel','placements','positions','profession','professional','professions','qualified','receptionist','recruit','recruiter','recruiters','recruiting','recruitment','resume','resumes','salaried','salaries','salary','seeking','skilled','staff','staffing','supervisor','trainee','vacancies','vacancy','worker','workers','workforce','workplace']
#
# SECURITY_SPAM_ALERTS= ['abducted','accidental','anthrax','anti','antibiotic','antibiotics','assaulted','attacked','attacker','attackers','auth','authenticated','authentication','avoid','avoidance','avoided','avoiding','bacteria','besieged','biometric','bioterrorism','blocking','boarded','bodyguards','botched','captive','captives','captors','captured','chased','commandeered','compromised','confronted','contagious','cornered','culprit','damage','damaging','danger','dangerous','dangers','destroying','destructive','deterrent','detrimental','disruptive','electrocuted','eliminate','eliminating','encroachment','encrypted','encryption','epidemic','escape','escaped','escapee','escaping','expose','exposed','exposing','fatally','feared','fled','flee','fleeing','flu','foiled','freed','germ','germs','guarded','guarding','guards','gunning','hapless','harassed','harm','harmful','harmless','harsh','hepatitis','hid','hijacked','hijacker','hijackers','hiv','hostage','hostages','hunted','immune','immunity','immunization','imprisoned','improper','inadvertent','infect','infected','infecting','infection','infections','infectious','infects','injuring','intentional','interference','interfering','intruders','intrusion','intrusive','invaded','isolates','kidnapped','limiting','login','logins','logon','lured','malaria','malicious','masked','minimise','minimize','minimizing','misuse','mite','mitigating','mosquito','motorcade','nuisance','offending','outbreak','overrun','passcode','password','passwords','plaintext','pneumonia','policeman','potentially','prevent','prevented','preventing','prevents','prone','protect','protected','protecting','protection','protects','quarantine','raided','ransom','raped','refuge','removing','rescued','rescuing','resisting','risks','robbed','runaway','safeguard','secret','secrets','seized','sensitive','server','shielding','smallpox','spam','spores','stolen','stormed','strain','strains','stranded','strep','summoned','susceptible','swine','threat','threatened','threatening','threats','thwarted','tortured','trapped','unaccounted','undesirable','unhealthy','unidentified','unintended','unintentional','unnamed','unnecessary','unprotected','unsafe','unwanted','unwelcome','user','username','vaccine','vaccines','villagers','viral','virus','viruses','vulnerability','vulnerable','whereabouts','whooping','withstand','wounded']
#
# SUPPORT=['ability','acrobat','adobe','advantage','advice','aid','aids','aim','alternatives','app','apps','ares','assist','autodesk','avs','benefits','best','boost','bring','bringing','build','cad','ccna','cellphone','challenge','choices','choosing','citrix','compatible','computer','computers','conferencing','console','consoles','continue','contribute','corel','create','creating','crucial','desktop','desktops','develop','devices','digital','discover','discuss','ease','easier','educate','effective','effectively','effort','electronic','electronics','encarta','encourage','energy','enhance','ensure','essential','eudora','experience','explore','finding','future','gadget','gadgets','gizmos','goal','groupwise','guide','guides','handhelds','handset','handsets','hardware','help','helpful','helping','helps','hopes','ideas','idm','important','improve','interactive','internet','introduce','intuit','invaluable','ios','join','kiosk','kiosks','laptops','lead','learn','lightwave','mac','machines','macintosh','macromedia','maintain','manage','mcafee','mcse','meet','messaging','metastock','microsoft','mobile','monitors','morpheus','mouse','mice','msie','multimedia','natural','needed','needs','netware','networked','networking','norton','notebooks','novell','ocr','oem','offline','office','opportunity','our','peripherals','personal','pgp','phone','phones','photoshop','plan','plans','portables','potential','practical','prepare','pros','quark','quicken','realplayer','recommend','remotely','resco','resources','safe','save','saving','sbe','screens','serve','servers','share','sharing','software','solve','sophos','spb','spss','ssg','standalone','support','symantec','task','tech','telephones','televisions','their','tips','to','together','trojan','useful','users','valuable','veritas','virtual','visio','vista','vital','vmware','ways','wga','whs','winzip','wordperfect','working','workstation','workstations','xp','xpress']
#
# ENERGY_DESK=['amps','baseload','bhp','biomass','blowers','boiler','boilers','btu','btus','burners','cc','cfm','chiller','chillers','cogen','cogeneration','compressors','conditioner','conditioners','conditioning','coolers','cooling','cranking','desalination','diesels','electric','electrical','electricity','electricty','electrification','energy','engine','engines','furnace','furnaces','gasification','generators','genset','geothermal','gigawatt','gpm','heat','heater','heaters','heating','horsepower','hp','hvac','hydro','hydroelectric','hydroelectricity','hydropower','idle','idling','ignition','interconnectors','intertie','kilovolt','kilowatt','kilowatts','kw','kwh','levelized','liter','megawatt','megawatts','microturbine','microturbines','motor','motors','mph','municipally','peaker','photovoltaic','photovoltaics','power','powered','powerplant','powerplants','psi','psig','reactors','redline','refrigerated','refrigeration','renewable','renewables','repower','repowering','retrofits','retrofitting','revs','rpm','siting','solar','substation','substations','switchgear','switchyard','temperatures','terawatt','thermo','thermoelectric','thermostat','thermostats','throttle','torque','turbine','turbines','turbo','undergrounding','ventilation','volt','volts','weatherization','whp','wind','windmill','windmills','windpower']
#
# SALES_DEPARTMENT=['accounting','actuals','advertised','affordable','auction','auctions','audited','auditing','bargain','bargains','bidding','billable','billed','billing','billings','bookkeeping','bought','brand','branded','brands','broker','brokerage','brokers','budgeting','bulk','buy','buyer','buyers','buying','buys','cancel','cancellation','cancellations','cancelled','cardholders','cashback','cashflow','chain','chargeback','chargebacks','cheap','cheaper','cheapest','checkbook','checkout','cheque','cheques','clearance','closeout','consignment','convenience','cosmetics','coupon','coupons','deals','debit','debited','debits','deducted','delivery','deposit','discontinued','discount','discounted','discounts','distributor','ebay','escrow','expensive','export','exported','exporter','exporters','exporting','exports','fee','fees','goods','gratuities','gratuity','groceries','grocery','import','importation','imported','importer','importers','importing','imports','incur','inexpensive','instore','inventory','invoice','invoiced','invoices','invoicing','item','items','lease','ledger','ledgers','manufacturer','marketed','merchandise','merchant','negotiable','nonmembers','nonrefundable','ordering','origination','outlets','overage','overdraft','overstock','owner','owners','payable','payables','payment','payroll','postage','postmarked','premium','prepaid','prepay','prepayment','price','priced','prices','pricey','pricing','product','products','proforma','purchase','purchased','purchaser','purchases','purchasing','rebate','rebook','rebooked','rebooking','receipts','receivable','receivables','reconciliations','recordkeeping','redeem','redeemable','refund','refundable','refunded','refunding','refunds','remittance','resell','reselling','retail','retailer','retailing','sale','sell','seller','sellers','selling','sells','shipment','shipments','shipped','shipper','shippers','shipping','shop','shopped','shopping','shops','sold','spreadsheets','store','stores','submittals','supermarket','supermarkets','superstore','supplier','supplies','supply','surcharge','surcharges','timesheet','timesheets','transaction','upfront','vending','vendor','verifications','voucher','vouchers','warehouse','warehouses','wholesale','wholesaler','wholesaling']
#LeadingNetworking
#LeadingRestraining
#LeadingFacilitating
#LeadingMotivating
#LeadingPhysicalAction
#LeadingPraise
#LeadingCriticism
#ThinkingLearningOrCreative
#ThinkingSpiritual
#ThinkingNobility
#ThinkingAmbiguity
#ThinkingCurrentAffairs
#ThinkingAnalytical
#SpeakingFormality
#SpeakingPop
#SpeakingGeek
#SpeakingCasualAndFamily
#SpeakingMachismo
#SpeakingHumanity
#SpeakingDramatic
#SpeakingBanter
#ActingUrgency
#ActingIndustryJargon
#ActingOfficialeseAndLegalese
#ActingTechSpeak
#ActingProjectManagement

#LEADING

LeadingNetworking=['accountants', 'administrators', 'advisers', 'advisors', 'alums', 'analysts', 'appraisers', 'architects', 'arrangers', 'assessors', 'assistants', 'attendants', 'auditors', 'backers', 'bankers', 'barbers', 'broadcasters', 'bureaucrats', 'bureaus', 'businessmen', 'businesspeople', 'businesswomen', 'campaigners', 'changers', 'clerks', 'coders', 'collaborators', 'collectors', 'columnists', 'commentators', 'conservationists', 'contributors', 'coordinators', 'correspondents', 'counselors', 'creatives', 'creators', 'dealmakers', 'designers', 'directors', 'dispatchers', 'doers', 'economists', 'educators', 'engineers', 'entertainers', 'entrepreneurs', 'environmentalists', 'examiners', 'execs', 'executives', 'exhibitors', 'facilitators', 'financiers', 'funders', 'generalists', 'geologists', 'greeters', 'gurus', 'handlers', 'helpers', 'historians', 'innovators', 'insiders', 'inspectors', 'integrators', 'intermediaries', 'interns', 'interviewees', 'interviewers', 'inventors', 'invitees', 'janitors', 'journalists', 'keepers', 'landowners', 'librarians', 'lobbyists', 'makers', 'managers', 'mangers', 'marketers', 'mediators', 'mentors', 'meteorologists', 'moderators', 'moguls', 'negotiators', 'newsroom', 'observers', 'operators', 'orgs', 'originators', 'paralegals', 'pickers', 'planners', 'policymakers', 'politicos', 'practitioners', 'principals', 'programmers', 'promoters', 'publicans', 'publics', 'pundits', 'raters', 'realtors', 'recruiters', 'regulators', 'reporters', 'reps', 'reviewers', 'salespeople', 'schedulers', 'screeners', 'secretaries', 'setters', 'signers', 'spokespeople', 'spokespersons', 'staffers', 'strategists', 'subcontractors', 'supervisors', 'surveyed', 'takers', 'tasters', 'techies', 'technologists', 'techs', 'telemarketers', 'testers', 'thinkers', 'treasurers', 'visionaries', 'watchdogs', 'watchers']

LeadingRestraining=['abandoning', 'adapting', 'adopting', 'aggregating', 'appreciating', 'assembling', 'auctioning', 'balking', 'beefing', 'bolstering', 'broadening', 'brokering', 'bundling', 'capitalizing', 'centralizing', 'championing', 'clarifying', 'commandeering', 'complicating', 'consolidating', 'contemplating', 'curtailing', 'dedicating', 'deliberating', 'devising', 'devoting', 'discounting', 'dismantling', 'distancing', 'diversifying', 'divesting', 'downgrading', 'downsizing', 'duplicating', 'elaborating', 'embarking', 'enlisting', 'envisioning', 'eschewing', 'expediting', 'experimenting', 'exploiting', 'fashioning', 'finalising', 'finalizing', 'flagging', 'fleshing', 'focussing', 'forging', 'forgoing', 'formalizing', 'formulating', 'groundwork', 'gutting', 'harmonization', 'harmonizing', 'harnessing', 'instituting', 'inventing', 'liberalizing', 'maturing', 'merging', 'mobilizing', 'modernizing', 'monetizing', 'mulling', 'navigating', 'optimising', 'orchestrating', 'overhauling', 'perfecting', 'personalizing', 'persuading', 'phasing', 'populating', 'previewing', 'prioritizing', 'procuring', 'rationalizing', 'readying', 'reaffirming', 'reassessing', 'reclaiming', 'reconciling', 'reconsidering', 'recreating', 'rectifying', 'redeeming', 'redefining', 'redrafting', 'reentering', 'reestablishing', 'reforming', 'regaining', 'reinstating', 'reinventing', 'relinquishing', 'renaming', 'renegotiating', 'renewing', 'reorganizing', 'repositioning', 'reserving', 'reshaping', 'restating', 'rethinking', 'retooling', 'revamping', 'reverting', 'revising', 'revisiting', 'revitalizing', 'reviving', 'reworking', 'rewriting', 'scrapping', 'scrutinizing', 'segmenting', 'shoring', 'shying', 'simplifying', 'solidifying', 'standardizing', 'streamlining', 'stressing', 'supplementing', 'tinkering', 'transfering', 'transitioning', 'tweaking', 'uncovering', 'underscoring', 'uniting', 'validating', 'valuing', 'visualizing', 'zeroing']

LeadingFacilitating=['able', 'accept', 'add', 'advantage', 'advice', 'advise', 'afford', 'all', 'allow', 'allowed', 'allowing', 'any', 'anytime', 'anywhere', 'are', 'arrange', 'ask', 'attempt', 'attempting', 'avoid', 'aware', 'be', 'become', 'begin', 'best', 'better', 'bring', 'build', 'call', 'calling', 'can', 'cannot', 'capture', 'careful', 'carry', 'chance', 'chances', 'change', 'changing', 'check', 'choice', 'choices', 'choose', 'choosing', 'clear', 'collect', 'come', 'compare', 'consider', 'continue', 'correct', 'create', 'deal', 'decide', 'deciding', 'different', 'difficult', 'discover', 'discuss', 'do', 'easier', 'easiest', 'easily', 'easy', 'either', 'enter', 'everyday', 'exact', 'expect', 'express', 'extra', 'fail', 'fair', 'fill', 'find', 'finding', 'follow', 'gather', 'get', 'give', 'giving', 'go', 'have', 'help', 'here', 'hold', 'if', 'important', 'instead', 'interested', 'introduce', 'job', 'keep', 'keeping', 'learn', 'leave', 'let', 'lets', 'likely', 'live', 'll', 'locate', 'longer', 'look', 'looking', 'lose', 'make', 'making', 'manage', 'many', 'may', 'meet', 'money', 'more', 'most', 'must', 'necessary', 'need', 'needed', 'needing', 'needs', 'no', 'not', 'notice', 'ones', 'opt', 'option', 'options', 'or', 'order', 'other', 'others', 'our', 'ourselves', 'own', 'people', 'perform', 'person', 'personal', 'pick', 'place', 'places', 'please', 'possible', 'prefer', 'prepare', 'promise', 'proper', 'put', 'quickly', 'ready', 'real', 'reasons', 'recognize', 'recommend', 'regularly', 'rely', 'repeat', 'replace', 'require', 'respond', 'rid', 'satisfied', 'save', 'saving', 'see', 'seek', 'serve', 'settle', 'share', 'should', 'simple', 'simply', 'solve', 'some', 'speak', 'spend', 'start', 'stay', 'suggest', 'take', 'taking', 'tend', 'their', 'them', 'themselves', 'these', 'they', 'those', 'tips', 'to', 'together', 'trust', 'try', 'unable', 'unless', 'us', 'use', 'using', 'visit', 'want', 'way', 'ways', 'we', 'well', 'whenever', 'wherever', 'whether', 'will', 'willing', 'wish', 'without', 'work', 'you', 'your', 'yourself']

LeadingMotivating=['abilities', 'accomplish', 'accomplished', 'accomplishing', 'accomplishments', 'achievement', 'achievements', 'actively', 'addressing', 'advancement', 'advancing', 'aim', 'aimed', 'aiming', 'aims', 'ambitions', 'approach', 'approaches', 'aspects', 'aspirations', 'attention', 'attract', 'attracting', 'avenues', 'awareness', 'becoming', 'beyond', 'blueprint', 'branding', 'bringing', 'broader', 'career', 'careers', 'challenge', 'challenges', 'challenging', 'commitment', 'committed', 'communicate', 'communicating', 'competitive', 'conducive', 'continuing', 'contribute', 'contributing', 'creating', 'creation', 'creative', 'creativity', 'critical', 'crucial', 'cultivating', 'demands', 'demonstrate', 'develop', 'developed', 'developing', 'development', 'devoted', 'discovering', 'diverse', 'diversity', 'driven', 'educate', 'educating', 'effort', 'embracing', 'emphasis', 'empowering', 'empowerment', 'encourage', 'encouraged', 'encouragement', 'encourages', 'encouraging', 'endeavor', 'endeavors', 'engage', 'engaged', 'engagement', 'engaging', 'enriching', 'entrepreneurial', 'evolving', 'expanding', 'expectations', 'experience', 'experiences', 'exploration', 'explore', 'exploring', 'focus', 'focused', 'focuses', 'focusing', 'focussed', 'formative', 'foster', 'fostered', 'fostering', 'fosters', 'friendships', 'fruitful', 'fulfill', 'fulfilling', 'furthering', 'future', 'gaining', 'geared', 'generation', 'goal', 'goals', 'growing', 'guiding', 'habits', 'helping', 'horizons', 'ideas', 'importance', 'importantly', 'increasingly', 'influence', 'influencing', 'innovative', 'insight', 'insights', 'inspire', 'interests', 'introducing', 'invaluable', 'involved', 'issues', 'knowledge', 'leadership', 'lifelong', 'lifestyle', 'lifestyles', 'meaningful', 'mentoring', 'milestones', 'mindset', 'motivate', 'motivated', 'motivating', 'motivation', 'niche', 'nontraditional', 'nurturing', 'opportunities', 'opportunity', 'parenting', 'peers', 'perspective', 'positive', 'possibilities', 'potential', 'practical', 'practices', 'productive', 'progress', 'progressing', 'promote', 'promoting', 'prospects', 'pursue', 'pursuing', 'pursuit', 'pursuits', 'raising', 'recognition', 'recognizing', 'relationship', 'relationships', 'researching', 'resources', 'reward', 'rewarding', 'rewards', 'role', 'roles', 'seeking', 'shaping', 'shared', 'sharing', 'social', 'solving', 'stimulating', 'strategies', 'strategy', 'strengths', 'strive', 'strives', 'striving', 'strong', 'succeed', 'success', 'successes', 'successful', 'tackling', 'talent', 'talents', 'tangible', 'targeted', 'targets', 'thrive', 'toward', 'towards', 'transformation', 'transforming', 'ultimately', 'understanding', 'valuable', 'viable', 'vision', 'vital', 'wealth', 'workplace', 'worthwhile', 'youth']

LeadingPhysicalAction=['across', 'ahead', 'aisle', 'along', 'apart', 'approaching', 'arms', 'around', 'away', 'back', 'backing', 'backs', 'backward', 'backwards', 'ball', 'balls', 'behind', 'bell', 'bend', 'big', 'bite', 'blast', 'block', 'blocks', 'bottom', 'bounce', 'break', 'breaking', 'broken', 'bump', 'bust', 'catch', 'catching', 'caught', 'chunk', 'circle', 'circles', 'close', 'closer', 'closest', 'corner', 'corners', 'counter', 'crawl', 'cross', 'cue', 'cut', 'cuts', 'deep', 'dig', 'direction', 'directions', 'ditch', 'down', 'drag', 'draw', 'drive', 'drop', 'dropping', 'dump', 'edge', 'empty', 'ends', 'face', 'faces', 'facing', 'fall', 'falling', 'farther', 'fast', 'finish', 'flag', 'flip', 'foot', 'fore', 'forward', 'front', 'gap', 'grab', 'ground', 'half', 'halfway', 'hand', 'hands', 'hang', 'hanging', 'haul', 'head', 'heading', 'heads', 'heavy', 'hide', 'hit', 'holding', 'hole', 'holes', 'hooked', 'huge', 'inside', 'into', 'jog', 'jump', 'keeper', 'kick', 'knock', 'lap', 'lay', 'lean', 'leap', 'leaving', 'left', 'line', 'lines', 'long', 'loose', 'mid', 'middle', 'midway', 'minute', 'move', 'moving', 'narrow', 'notch', 'off', 'onto', 'open', 'opposite', 'out', 'outside', 'pace', 'pass', 'passing', 'path', 'pause', 'pinch', 'pit', 'pits', 'point', 'pointing', 'pose', 'pull', 'punch', 'push', 'quick', 'reach', 'reaching', 'rest', 'resting', 'reverse', 'right', 'roll', 'rolling', 'rough', 'round', 'row', 'run', 'running', 'rush', 'scramble', 'seconds', 'shake', 'sharp', 'shot', 'shots', 'shut', 'side', 'sides', 'sight', 'sit', 'sitting', 'skip', 'slack', 'slide', 'slides', 'slightly', 'slip', 'slow', 'slowly', 'snag', 'spin', 'split', 'spot', 'spots', 'spread', 'squeeze', 'stall', 'stand', 'standing', 'steady', 'step', 'stepping', 'steps', 'stick', 'stop', 'stopping', 'straight', 'stride', 'stuck', 'sweep', 'swing', 'tap', 'then', 'throw', 'thru', 'tick', 'tied', 'till', 'tiny', 'tip', 'top', 'toss', 'touch', 'tough', 'trailing', 'trap', 'triangle', 'trick', 'turn', 'turning', 'underneath', 'up', 'upside', 'walk', 'walking', 'whilst', 'zone', 'bare', 'baring', 'bellies', 'bending', 'bent', 'biting', 'blasting', 'blowing', 'bouncing', 'bowing', 'bracing', 'bucking', 'bumping', 'burly', 'busily', 'busting', 'carving', 'chew', 'choke', 'choking', 'chomping', 'chopping', 'circling', 'claw', 'claws', 'clinging', 'clipping', 'cracking', 'cranking', 'crawling', 'crouching', 'crunching', 'crushing', 'cutting', 'digging', 'dragging', 'erect', 'eyeballs', 'finger', 'fingers', 'fist', 'flashing', 'flipping', 'fours', 'furiously', 'gash', 'grabbing', 'grappling', 'grind', 'grinding', 'habit', 'hammer', 'hammering', 'hammers', 'handing', 'hauling', 'hilt', 'hooking', 'hopping', 'hugging', 'hump', 'hustling', 'jab', 'jabs', 'jacking', 'juggling', 'jumping', 'kicking', 'knocking', 'laying', 'leaning', 'leaping', 'lifting', 'limp', 'looping', 'milking', 'mouths', 'nailing', 'necks', 'nip', 'noses', 'parting', 'picking', 'piercing', 'piling', 'pinching', 'pointy', 'poke', 'poking', 'popping', 'pounding', 'pressing', 'probing', 'prodding', 'pry', 'pulling', 'punching', 'pushing', 'racking', 'raking', 'reins', 'ripping', 'rocking', 'rooting', 'ropes', 'rounding', 'rubbing', 'sagging', 'scrambling', 'scraping', 'scratching', 'shaking', 'sharpened', 'shovel', 'shovels', 'shoving', 'shredding', 'shreds', 'sideways', 'skipping', 'slamming', 'slapping', 'slashing', 'slippery', 'slipping', 'smashing', 'snapping', 'spinning', 'splitting', 'spotting', 'squat', 'squeezing', 'stab', 'sticking', 'stiff', 'straddling', 'straining', 'stretching', 'strokes', 'stump', 'swaying', 'swinging', 'swipe', 'taping', 'tapping', 'tearing', 'throwing', 'thrust', 'tipping', 'tongues', 'tossing', 'touching', 'trimming', 'tripping', 'twisting', 'tying', 'urinating', 'wagging', 'wheeling', 'whip', 'whipping', 'wiggle', 'wiping', 'wobbly', 'wringing', 'yank','aces', 'against', 'battled', 'beat', 'beating', 'berth', 'birdie', 'bogey', 'bogeys', 'breakaway', 'champ', 'champion', 'champions', 'championship', 'championships', 'champs', 'clash', 'clinch', 'clinching', 'comeback', 'compete', 'competed', 'competes', 'competing', 'competition', 'competitor', 'competitors', 'consecutive', 'consolation', 'contender', 'contenders', 'contest', 'contestant', 'contestants', 'contested', 'contests', 'decisive', 'defeat', 'defeated', 'defeating', 'defeats', 'defending', 'derby', 'discus', 'division', 'divisional', 'dominated', 'dominating', 'doubles', 'earned', 'eighth', 'eventual', 'fielded', 'fifth', 'final', 'finalist', 'finalists', 'finals', 'finishes', 'finishing', 'fourth', 'handicap', 'handily', 'heavyweight', 'hurdle', 'invitational', 'javelin', 'lone', 'longshot', 'losing', 'match', 'matches', 'matchup', 'matchups', 'medal', 'medals', 'narrowly', 'ninth', 'opener', 'opponent', 'opponents', 'overcame', 'pairings', 'pennant', 'penultimate', 'pitted', 'playoff', 'podium', 'postseason', 'putt', 'qualifier', 'qualifiers', 'qualifying', 'race', 'races', 'ranked', 'ranking', 'rankings', 'reigning', 'rival', 'rivalry', 'rivals', 'rout', 'runner', 'runners', 'score', 'scores', 'seeded', 'semifinals', 'seventh', 'shootout', 'showdown', 'singles', 'sixth', 'slam', 'sprint', 'sprinters', 'standings', 'streak', 'tally', 'tiebreaker', 'title', 'toughest', 'tournament', 'tourney', 'triumph', 'trophy', 'unbeaten', 'undefeated', 'underdog', 'upsets', 'victories', 'victory', 'vs', 'win', 'winner', 'winners', 'winning', 'wins', 'won','ably', 'abundantly', 'acutely', 'adequately', 'aggressively', 'alternately', 'amicably', 'amply', 'appropriately', 'arbitrarily', 'artificially', 'attentively', 'attractively', 'bilaterally', 'boldly', 'broadly', 'carefully', 'cautiously', 'channeled', 'cheaply', 'cleanly', 'cleverly', 'cohesive', 'comfortably', 'competitively', 'comprehensively', 'conceptually', 'concurrently', 'confidently', 'consciously', 'conservatively', 'consistently', 'conspicuously', 'constantly', 'constructively', 'continually', 'continuously', 'cooperatively', 'correspondingly', 'creatively', 'critically', 'decisively', 'deeply', 'diligently', 'distinctly', 'endlessly', 'energized', 'enormously', 'enthusiastically', 'evenly', 'excessively', 'expeditiously', 'extensively', 'faithfully', 'favorably', 'feverishly', 'fiercely', 'firmly', 'fleshed', 'forcefully', 'freely', 'gradually', 'handsomely', 'heavily', 'honed', 'hotly', 'immersed', 'incrementally', 'independently', 'ineffectively', 'inexpensively', 'informally', 'infrequently', 'intelligently', 'intensely', 'interestingly', 'intimately', 'intuitively', 'keenly', 'labored', 'legitimately', 'liberally', 'loosely', 'massively', 'matured', 'mightily', 'moderately', 'modestly', 'nimbly', 'optimally', 'ostensibly', 'passively', 'persistently', 'piecemeal', 'positively', 'predictably', 'proactively', 'productively', 'profitably', 'progressively', 'prudently', 'purposefully', 'radically', 'rapidly', 'rationally', 'realistically', 'relentlessly', 'reliably', 'remarkably', 'respectfully', 'responsibly', 'restrained', 'rigorously', 'routinely', 'satisfactorily', 'scrutinized', 'sensibly', 'simultaneously', 'smoothly', 'solidly', 'sparingly', 'squarely', 'strategically', 'subtly', 'sufficiently', 'suitably', 'swiftly', 'systematically', 'tended', 'thoroughly', 'thoughtfully', 'tightly', 'tirelessly', 'transparently', 'tremendously', 'uniformly', 'vastly', 'vigorously', 'vitally', 'weakly', 'wisely']

LeadingPraise=['acclaim', 'accolades', 'admired', 'aided', 'alongside', 'amalgamation', 'amassed', 'amassing', 'amongst', 'arguably', 'astounding', 'attracted', 'behemoth', 'bevy', 'biggest', 'boast', 'boasted', 'boasting', 'boldest', 'bonafide', 'boon', 'breakthrough', 'brightest', 'broadest', 'budding', 'burgeoning', 'clout', 'colossal', 'cornerstone', 'counterpart', 'counterparts', 'coveted', 'culmination', 'cyberspace', 'decade', 'destined', 'distinguished', 'dominance', 'dominate', 'eighties', 'eleventh', 'elite', 'elusive', 'emerge', 'endowed', 'enormous', 'envisioned', 'erstwhile', 'esteemed', 'fabled', 'fame', 'famed', 'famous', 'favored', 'favoured', 'flagship', 'fledged', 'fledgling', 'flourish', 'foothold', 'foray', 'forays', 'forefront', 'foremost', 'formidable', 'fortunes', 'franchises', 'fruition', 'gargantuan', 'garner', 'garnered', 'garnering', 'geniuses', 'giants', 'globe', 'greatest', 'groundbreaking', 'hailed', 'heavyweights', 'helm', 'heralded', 'homage', 'homegrown', 'hugely', 'illustrious', 'imaginable', 'immense', 'inception', 'indispensable', 'influential', 'inroads', 'irreplaceable', 'landmark', 'launching', 'legacy', 'legendary', 'limelight', 'lofty', 'longstanding', 'lucrative', 'luminaries', 'mammoth', 'marketable', 'marvel', 'mecca', 'milestone', 'millionth', 'minted', 'moniker', 'monumental', 'myriad', 'newcomer', 'newcomers', 'newest', 'newfound', 'nineties', 'notable', 'notably', 'noteworthy', 'notoriety', 'overlooked', 'peerless', 'perfected', 'personalities', 'phenomenal', 'pinnacle', 'pioneered', 'pioneering', 'pioneers', 'plethora', 'poised', 'popular', 'popularity', 'powerhouse', 'powerhouses', 'predecessor', 'predecessors', 'preeminent', 'prestige', 'prime', 'prized', 'prolific', 'prominence', 'prominent', 'promising', 'prospect', 'prowess', 'ranks', 'reckoned', 'renowned', 'reputation', 'reputations', 'revolutionized', 'rewarded', 'richest', 'rumored', 'shortlist', 'sizable', 'sizeable', 'slew', 'spearhead', 'spotlight', 'springboard', 'stateside', 'stature', 'stellar', 'stints', 'strides', 'strongest', 'succeeding', 'superpower', 'superstars', 'surpass', 'surpasses', 'surpassing', 'synonymous', 'teamed', 'teaming', 'testament', 'thrived', 'thriving', 'touted', 'touting', 'tremendous', 'ubiquitous', 'unbeatable', 'undiscovered', 'undisputed', 'undoubtedly', 'unequaled', 'unheard', 'unofficial', 'unparalleled', 'unprecedented', 'unrivaled', 'unsung', 'untapped', 'unveil', 'unveiled', 'unveiling', 'upstart', 'vanguard', 'vast', 'vaunted', 'venerable', 'vying', 'workhorse', 'worthy']

LeadingCriticism=['abnormally', 'abrupt', 'absence', 'accelerated', 'accidental', 'accumulate', 'accumulation', 'adverse', 'adversely', 'affect', 'affected', 'affecting', 'affects', 'aging', 'alteration', 'alterations', 'apparent', 'associated', 'attributable', 'attributed', 'attrition', 'avoidance', 'avoided', 'behavior', 'behaviour', 'bodily', 'breakage', 'breakdown', 'cause', 'caused', 'causes', 'causing', 'cessation', 'changes', 'circulation', 'compensate', 'compensatory', 'concurrent', 'condition', 'conditions', 'consequence', 'consequences', 'consequent', 'continual', 'contraction', 'damage', 'decay', 'decrease', 'decreased', 'decreases', 'decreasing', 'defect', 'defective', 'defects', 'deficits', 'dependence', 'depletion', 'deterioration', 'difficulty', 'diminished', 'disrupted', 'disruption', 'disturbance', 'diversion', 'drastic', 'due', 'effect', 'effected', 'effects', 'elevated', 'elimination', 'erosion', 'exacerbated', 'excessive', 'exhaustion', 'experiencing', 'exposure', 'exposures', 'extreme', 'factors', 'failure', 'fluctuating', 'fluctuations', 'frequent', 'gradual', 'heightened', 'imbalance', 'immediate', 'impacted', 'impaired', 'impairment', 'impairments', 'inability', 'inactivity', 'indication', 'indications', 'indicative', 'indirect', 'infancy', 'inflow', 'infrequent', 'instability', 'insufficient', 'intermittent', 'interruption', 'involuntary', 'irregular', 'lack', 'likelihood', 'loss', 'malfunction', 'minor', 'moderate', 'mortality', 'negative', 'negatively', 'neglect', 'negligible', 'normal', 'noticeable', 'obstruction', 'occur', 'occuring', 'occurrence', 'occurrences', 'occurring', 'occurs', 'outflow', 'overload', 'overuse', 'owing', 'partial', 'periods', 'persist', 'persistent', 'persists', 'physical', 'potentially', 'premature', 'pressures', 'problems', 'prolonged', 'prone', 'propensity', 'rapid', 'recurring', 'reduced', 'reduction', 'relief', 'repeated', 'result', 'resulted', 'resulting', 'reversal', 'risk', 'risks', 'secondary', 'severely', 'severity', 'shrinkage', 'significant', 'slight', 'sporadic', 'stagnation', 'stress', 'stresses', 'subsequent', 'sudden', 'susceptible', 'sustained', 'systemic', 'temporary', 'tendency', 'tolerated', 'transient', 'trigger', 'triggered', 'triggering', 'triggers', 'unaffected', 'uncontrollable', 'uncontrolled', 'undetermined', 'unexplained', 'unfavorable', 'unspecified', 'weakness', 'widespread', 'withdrawal', 'worsen', 'worsening','acrimonious', 'afloat', 'ailing', 'ails', 'awry', 'backsliding', 'beleaguered', 'blunder', 'bogged', 'brink', 'buffeted', 'bungled', 'bungling', 'chalked', 'clouded', 'crippled', 'crumbling', 'dangerously', 'deadlock', 'decimated', 'derailed', 'desparately', 'deteriorating', 'dicey', 'disarray', 'dogged', 'doldrums', 'doomed', 'eke', 'embroiled', 'extricate', 'falter', 'faltered', 'faltering', 'fated', 'floundering', 'footing', 'foundering', 'fragile', 'fray', 'fumbling', 'gambit', 'gridlock', 'haggling', 'hampered', 'handedly', 'hardball', 'haywire', 'headway', 'hemorrhaging', 'hindered', 'hobbled', 'hopelessly', 'imbroglio', 'impasse', 'impenetrable', 'implode', 'imploded', 'irreparably', 'jeopardized', 'jeopardy', 'jockeying', 'languish', 'languished', 'languishing', 'lifeline', 'limbo', 'limping', 'linchpin', 'littered', 'logjam', 'loomed', 'maneuverings', 'marred', 'materialise', 'mired', 'morass', 'moribund', 'muddle', 'nary', 'overburdened', 'overheated', 'panacea', 'paralyzed', 'pariah', 'perilously', 'politicking', 'precarious', 'precipice', 'prematurely', 'quagmire', 'reeling', 'riddled', 'roadblock', 'roiled', 'rut', 'saddled', 'scuttle', 'scuttled', 'shambles', 'sidelined', 'snafu', 'soured', 'souring', 'squabbling', 'stalemate', 'stalled', 'stalling', 'stave', 'stifled', 'stifling', 'straits', 'stubbornly', 'stumbling', 'stymied', 'suitors', 'swamped', 'tailspin', 'tangle', 'tangled', 'tarnished', 'tatters', 'teetering', 'tenuous', 'thorny', 'thwarted', 'tightrope', 'topple', 'torrid', 'tragically', 'treading', 'tussle', 'uncharted', 'undoing', 'undone', 'unfulfilled', 'unnoticed', 'unravel', 'unraveled', 'unraveling', 'unscathed', 'verge', 'wayside', 'wither', 'withering', 'withstood', 'wrangle', 'wrangles', 'wrangling']



#THINKING

ThinkingLearningOrCreative=['accompanies', 'accompanying', 'alternates', 'articulated', 'captures', 'capturing', 'centered', 'characterised', 'characterizes', 'compares', 'concludes', 'contrast', 'contrasted', 'contrasts', 'convey', 'conveyed', 'conveys', 'demonstrates', 'demonstrating', 'depict', 'depicted', 'depicting', 'depiction', 'depicts', 'describe', 'described', 'describes', 'describing', 'detail', 'detailing', 'diagrammatic', 'discusses', 'documenting', 'drawn', 'elaborated', 'emphasise', 'emphasised', 'emphasize', 'emphasized', 'emphasizes', 'emphasizing', 'encompassing', 'examines', 'exemplary', 'exemplified', 'exemplifies', 'exhibited', 'exhibiting', 'exhibits', 'explored', 'explores', 'exposition', 'expresses', 'facets', 'figures', 'gleaned', 'graphically', 'highlight', 'highlighted', 'highlighting', 'highlights', 'illuminate', 'illuminating', 'illustrate', 'illustrated', 'illustrates', 'illustrating', 'illustrative', 'imagery', 'incorporating', 'infographic', 'instructive', 'interspersed', 'intertwined', 'interwoven', 'modeled', 'modelled', 'narrative', 'outline', 'outlines', 'outlining', 'parallels', 'passages', 'pivotal', 'plots', 'portray', 'portrayed', 'portraying', 'portrays', 'presented', 'presenting', 'presents', 'profiled', 'prominently', 'recalling', 'recounting', 'recounts', 'reflect', 'reflected', 'reflecting', 'reflections', 'reflects', 'relates', 'resemblance', 'resembles', 'retrospective', 'reveal', 'revealing', 'revolved', 'shadowing', 'sketched', 'spotlights', 'summarised', 'summarized', 'summarizes', 'summarizing', 'summed', 'themes', 'traces', 'tracing', 'underlines', 'underscore', 'underscored', 'underscores', 'unfolded', 'vignette', 'vignettes', 'visual', 'visually', 'vividly', 'absorb', 'actualize', 'adapt', 'align', 'alter', 'anew', 'assemble', 'assimilate', 'augment', 'bolster', 'brainstorm', 'broaden', 'capitalize', 'carve', 'categorize', 'centralize', 'circulate', 'classify', 'coexist', 'collaborate', 'colonize', 'compose', 'compress', 'conserve', 'consolidate', 'converge', 'cultivate', 'decipher', 'deepen', 'develope', 'devise', 'differentiate', 'dismantle', 'disseminate', 'distribute', 'diverge', 'diversify', 'divide', 'downsize', 'elevate', 'empower', 'emulate', 'energize', 'enlarge', 'enrich', 'equip', 'evolve', 'expand', 'exploit', 'extend', 'familiarize', 'flatten', 'forge', 'formalize', 'formulate', 'fortify', 'hone', 'impart', 'incorporate', 'incubate', 'infuse', 'initiate', 'inject', 'innovate', 'interact', 'internalize', 'invent', 'isolate', 'juggle', 'jumpstart', 'lengthen', 'liberate', 'manipulate', 'meld', 'mend', 'merge', 'migrate', 'mobilize', 'modernize', 'modify', 'monetize', 'nurture', 'orchestrate', 'organise', 'organize', 'orient', 'orientate', 'pinpoint', 'preserve', 'prioritize', 'propel', 'publicize', 'reaffirm', 'realign', 'rearrange', 'reassess', 'recapture', 'reclaim', 'reconcile', 'reconstruct', 'recreate', 'rectify', 'redefine', 'redeploy', 'rediscover', 'redistribute', 'redouble', 'reestablish', 'reevaluate', 'refine', 'refocus', 'regain', 'regroup', 'rehabilitate', 'reinforce', 'reintroduce', 'reinvent', 'rejuvenate', 'rekindle', 'relearn', 'relocate', 'remediate', 'renew', 'reorganize', 'replenish', 'replicate', 'repopulate', 'reproduce', 'reshape', 'resolve', 'restructure', 'resurface', 'resurrect', 'retain', 'retake', 'rethink', 'retool', 'revise', 'revisit', 'revitalize', 'revive', 'revolutionize', 'rework', 'scour', 'segregate', 'sharpen', 'shorten', 'sift', 'socialize', 'solidify', 'stabilize', 'standardize', 'stimulate', 'straighten', 'strategize', 'strengthen', 'summarise', 'supercharge', 'transform', 'uncover', 'unify', 'unite', 'uplift', 'utilise', 'visualize', 'widen']

ThinkingSpiritual=['affection', 'astral', 'attraction', 'aura', 'awakened', 'awakening', 'awe', 'beings', 'bliss', 'companionship', 'conscience', 'conscious', 'consciousness', 'contemplation', 'cosmos', 'curiosity', 'darkest', 'darkness', 'deeper', 'deepest', 'desire', 'desires', 'destiny', 'divine', 'dream', 'dreaming', 'dreams', 'earth', 'ecstasy', 'ego', 'embrace', 'emotion', 'emotional', 'emotions', 'endless', 'energies', 'enigma', 'enlightenment', 'epiphany', 'essence', 'eternal', 'eternity', 'existence', 'fascination', 'feelings', 'fleeting', 'friendship', 'fulfillment', 'fullest', 'goodness', 'gratification', 'gratitude', 'greatness', 'happiness', 'harmony', 'healing', 'hidden', 'humanity', 'illusion', 'illusions', 'imaginary', 'imagination', 'imagined', 'impulse', 'indulgence', 'inexhaustible', 'infinite', 'innocence', 'inspiration', 'inspirations', 'instinct', 'instincts', 'intellect', 'intentions', 'intimacy', 'intuition', 'invisible', 'journey', 'joy', 'joys', 'karma', 'lies', 'life', 'limitless', 'lust', 'magic', 'magical', 'manifest', 'manifestation', 'mankind', 'mantra', 'meditation', 'meditations', 'memories', 'minds', 'muse', 'mysteries', 'mystery', 'mystic', 'myth', 'nature', 'obsession', 'oneself', 'passion', 'passions', 'perfection', 'perpetual', 'personality', 'pleasures', 'profound', 'psyche', 'psychic', 'pure', 'realisation', 'reality', 'realization', 'realm', 'recollection', 'reflection', 'secrets', 'self', 'sense', 'senses', 'serendipity', 'sexuality', 'soul', 'souls', 'spirit', 'spirits', 'spiritual', 'spirituality', 'thoughts', 'true', 'truth', 'truths', 'ultimate', 'universe', 'unseen', 'visions', 'wisdom', 'wonders', 'world', 'worlds']

ThinkingNobility=['abiding', 'acceptance', 'accomplishment', 'accountability', 'acumen', 'adherence', 'ambition', 'anonymity', 'appreciation', 'appropriateness', 'aptitude', 'assurance', 'attainment', 'attitude', 'attractiveness', 'authenticity', 'autonomy', 'breadth', 'camaraderie', 'candor', 'civility', 'clarity', 'committment', 'compassion', 'competence', 'completeness', 'composure', 'confidence', 'conformity', 'consistency', 'continuity', 'cornerstones', 'correctness', 'courage', 'credibility', 'decency', 'dedication', 'dependability', 'determination', 'dignity', 'diligence', 'diplomacy', 'discernment', 'enthusiasm', 'esteem', 'ethic', 'ethos', 'excellence', 'exclusivity', 'expectation', 'expediency', 'fairness', 'familiarity', 'fidelity', 'finesse', 'firmness', 'foresight', 'formality', 'generosity', 'goodwill', 'hallmark', 'honesty', 'humility', 'independence', 'ingenuity', 'instill', 'integrity', 'intelligence', 'journalistic', 'kindness', 'lacked', 'loyalty', 'manners', 'maturity', 'moderation', 'morale', 'motto', 'neutrality', 'normalcy', 'objectivity', 'openness', 'optimism', 'originality', 'paramount', 'patience', 'perseverance', 'persistence', 'persuasion', 'predictability', 'pride', 'professionalism', 'propriety', 'prudence', 'punctuality', 'purity', 'qualities', 'readiness', 'reassurance', 'reciprocity', 'refinement', 'regularity', 'resilience', 'resiliency', 'respect', 'restraint', 'sanctity', 'sanity', 'satisfaction', 'seriousness', 'simplicity', 'sincerity', 'smarts', 'soundness', 'sufficiency', 'superiority', 'tact', 'teamwork', 'tenacity', 'thoughtfulness', 'timeliness', 'tolerance', 'transparency', 'trump', 'trumps', 'unconditional', 'underpinned', 'uniformity', 'uniqueness', 'urgency', 'usefulness', 'utmost', 'vigilance', 'vigor', 'virtue', 'virtues', 'vitality', 'willingness', 'workmanship', 'worthiness', 'yardstick', 'zeal']

ThinkingAmbiguity=['abstract', 'ambiguity', 'ambiguous', 'analogies', 'analogy', 'assumptions', 'attitudes', 'beliefs', 'boundaries', 'categorization', 'causal', 'coherent', 'complexity', 'concept', 'conception', 'concepts', 'conceptual', 'conflicting', 'considerations', 'construct', 'constructs', 'context', 'contexts', 'continuum', 'contradictory', 'conventions', 'define', 'definition', 'definitional', 'definitions', 'demarcation', 'derive', 'dichotomy', 'differences', 'differing', 'discourse', 'disparate', 'distinction', 'distinctions', 'distinguishing', 'divergent', 'elaboration', 'embodied', 'esoteric', 'explanations', 'explicit', 'expressions', 'fundamental', 'fundamentally', 'gender', 'groupings', 'hierarchies', 'hierarchy', 'identities', 'identity', 'imperfect', 'implications', 'implicit', 'incompatible', 'influences', 'inherent', 'interplay', 'interpret', 'interpretation', 'interpretations', 'interpreted', 'interpreting', 'interrelated', 'intricacies', 'layman', 'literal', 'logic', 'logical', 'materiality', 'meanings', 'metaphor', 'metaphors', 'motivations', 'myths', 'norms', 'notion', 'notions', 'nuance', 'nuances', 'overarching', 'overriding', 'paradigm', 'paradox', 'peculiar', 'perception', 'perceptions', 'pervasive', 'phenomena', 'philosophical', 'philosophies', 'precedence', 'prescriptive', 'principle', 'principles', 'rational', 'realism', 'relate', 'relational', 'relevance', 'representation', 'representations', 'rhetorical', 'rooted', 'salient', 'semantics', 'significance', 'similarities', 'simplification', 'singular', 'standpoint', 'stereotypes', 'subjective', 'symbolic', 'tendencies', 'tenets', 'terminology', 'textual', 'theories', 'totality', 'transcend', 'underlie', 'underlying', 'understandings', 'universal', 'viewpoint', 'viewpoints', 'workings', 'worldview']

ThinkingCurrentAffairs=['abramoff', 'algore', 'ayers', 'baghdad', 'baucus', 'bloomberg', 'bnp', 'buffett', 'carlin', 'carville', 'cbs', 'cheney', 'cia', 'clegg', 'clinton', 'clintons', 'cnbc', 'cnn', 'colbert', 'condoleezza', 'conseco', 'corzine', 'coulter', 'crist', 'dalai', 'daschle', 'dionne', 'dnc', 'dodd', 'dubya', 'durbin', 'eisenhower', 'emanuel', 'enron', 'fannie', 'fbi', 'feingold', 'feinstein', 'fiorina', 'giuliani', 'goldman', 'gop', 'govenor', 'grassley', 'greenpeace', 'greenspan', 'greenwald', 'gwb', 'hagel', 'halliburton', 'heckler', 'hillary', 'hussein', 'imus', 'iwo', 'jeb', 'jfk', 'jima', 'kissinger', 'kristol', 'krugman', 'kwanzaa', 'ladin', 'landrieu', 'lehrer', 'leno', 'lewinsky', 'lieberman', 'limbaugh', 'maher', 'mccain', 'mcconnell', 'mideast', 'mlk', 'monsanto', 'msnbc', 'mugabe', 'murdoch', 'nader', 'nafta', 'nbc', 'ndp', 'newt', 'nightline', 'nixon', 'noam', 'nobel', 'nostradamus', 'npr', 'nra', 'osama', 'pbs', 'pelosi', 'pentagon', 'perot', 'presiden', 'putin', 'qaeda', 'quayle', 'rangel', 'rasmussen', 'reagan', 'reich', 'rnc', 'rockefeller', 'romney', 'rothschild', 'rove', 'rumsfeld', 'sachs', 'saddam', 'saudis', 'schumer', 'schwarzenegger', 'sharpton', 'soros', 'spitzer', 'stalin', 'thatcher', 'viewitem', 'wallstreet', 'watergate', 'waxman', 'whitehouse', 'winfrey', 'wtc', 'yorker']

ThinkingAnalytical=['adder', 'algorithm', 'algorithms', 'approximate', 'approximating', 'approximation', 'arbitrary', 'arithmetic', 'bifurcation', 'binomial', 'bounds', 'calculate', 'calculated', 'calculates', 'calculating', 'calculation', 'calculations', 'calculator', 'clustering', 'coefficients', 'computation', 'computations', 'compute', 'computed', 'constrained', 'constraint', 'convergence', 'convergent', 'coordinates', 'correlation', 'correlations', 'correlator', 'correspond', 'corresponds', 'covariance', 'decimal', 'decimals', 'denominator', 'denoted', 'denotes', 'derivation', 'deriving', 'determinant', 'deviation', 'deviations', 'differential', 'differentials', 'digit', 'digits', 'dimensional', 'discrete', 'distributions', 'divergence', 'epsilon', 'equation', 'equations', 'equilibrium', 'estimation', 'estimations', 'exponent', 'exponential', 'extrapolation', 'factored', 'factoring', 'finite', 'formulae', 'formulas', 'fractional', 'fractions', 'geometric', 'geometry', 'graph', 'graphs', 'increment', 'infinity', 'integer', 'interpolation', 'inverse', 'inversely', 'iterated', 'iteration', 'iterations', 'linear', 'logistic', 'magnitudes', 'mathematical', 'mathematically', 'matrices', 'matrix', 'maximization', 'metric', 'minimization', 'monad', 'multiples', 'multiplied', 'multiplier', 'multiply', 'multiplying', 'nodal', 'normalized', 'normals', 'numeric', 'numerical', 'numerics', 'ordinate', 'parity', 'percentile', 'permutations', 'plotted', 'probabilistic', 'probabilities', 'probability', 'proportional', 'radix', 'ratios', 'recalculated', 'reciprocal', 'regression', 'scalar', 'scalars', 'scaled', 'scaling', 'sequential', 'sigma', 'skew', 'skewness', 'solver', 'squared', 'statistic', 'stochastic', 'subgroup', 'subgroups', 'subset', 'substitution', 'subtract', 'subtracted', 'subtracting', 'summation', 'symmetry', 'theta', 'trinomial', 'truncated', 'unconstrained', 'values', 'variables', 'variance', 'variances', 'vector', 'vertex', 'weighted', 'weighting', 'zeros', 'acceptability', 'accurate', 'accurately', 'actionable', 'actuarial', 'adequacy', 'aggregated', 'analyse', 'analysed', 'analyses', 'analysing', 'analysis', 'analytic', 'analytical', 'analyze', 'analyzed', 'analyzing', 'applicability', 'appraisal', 'assess', 'assessed', 'assessing', 'assessment', 'assessments', 'baseline', 'baselines', 'benchmark', 'benchmarking', 'benchmarks', 'characterizing', 'charting', 'classification', 'classifications', 'comparing', 'comparison', 'comparisons', 'computerized', 'conducted', 'confirmatory', 'conformance', 'criteria', 'criterion', 'data', 'datasets', 'delineation', 'demographic', 'demographics', 'determine', 'determining', 'disaggregated', 'econometric', 'estimating', 'evaluate', 'evaluated', 'evaluating', 'evaluation', 'evaluations', 'examine', 'examined', 'examining', 'exploratory', 'feasibility', 'findings', 'fingerprinting', 'forecasting', 'geographic', 'guideline', 'identification', 'identify', 'identifying', 'indicators', 'investigate', 'labeling', 'mapping', 'measurable', 'measures', 'methodologies', 'methodology', 'methods', 'metrics', 'modeling', 'modelling', 'monitoring', 'objective', 'observation', 'observations', 'outcome', 'outcomes', 'predict', 'predicting', 'prediction', 'predictions', 'predictive', 'preliminary', 'prioritization', 'prioritized', 'profiling', 'prospective', 'prospectively', 'qualitative', 'quantifiable', 'quantification', 'quantified', 'quantify', 'quantifying', 'quantitative', 'questionnaire', 'questionnaires', 'respondents', 'results', 'sampling', 'scenarios', 'scoping', 'scorecard', 'segmentation', 'simulation', 'simulations', 'standardised', 'standardized', 'statistical', 'statistically', 'suitability', 'survey', 'surveying', 'surveys', 'systematic', 'test', 'testing', 'tests', 'validate', 'validated', 'validation']

#SPEAKING

SpeakingFormality=['absent', 'absolute', 'acceptable', 'accordingly', 'act', 'actions', 'acts', 'actual', 'additionally', 'adhere', 'advantageous', 'afforded', 'aforementioned', 'akin', 'alternate', 'alternative', 'alternatively', 'alternatives', 'altogether', 'amenable', 'analogous', 'appear', 'appearing', 'applied', 'applies', 'applying', 'appropriate', 'arises', 'aspect', 'assumed', 'attained', 'basis', 'belong', 'belongs', 'bodies', 'borne', 'bound', 'broad', 'cases', 'certain', 'chiefly', 'chosen', 'circumstance', 'circumstances', 'clearly', 'closely', 'collectively', 'common', 'commonly', 'comparable', 'comparatively', 'conceivable', 'concern', 'concerned', 'confined', 'confines', 'conform', 'consequently', 'considerable', 'consideration', 'considered', 'consist', 'conspicuous', 'constitutes', 'customary', 'dealing', 'dealt', 'defined', 'definite', 'depend', 'dependant', 'depended', 'dependent', 'depending', 'depends', 'desirable', 'determined', 'devised', 'dictated', 'dictates', 'differ', 'differed', 'difference', 'differently', 'differs', 'distinct', 'distinguish', 'dominant', 'eg', 'elsewhere', 'employed', 'employing', 'encompass', 'entail', 'entails', 'entirely', 'entirety', 'equally', 'essentially', 'evidenced', 'evident', 'evolved', 'except', 'exception', 'exceptions', 'exclude', 'excluded', 'exclusively', 'exist', 'exists', 'extent', 'fairly', 'favorable', 'favourable', 'feasible', 'formal', 'forms', 'forth', 'frequently', 'furthermore', 'general', 'generally', 'hence', 'however', 'identical', 'identifiable', 'identified', 'imperative', 'implies', 'inclusion', 'indicate', 'indicated', 'indicates', 'indicating', 'indirectly', 'individual', 'individuals', 'influenced', 'instance', 'instances', 'intended', 'intent', 'invariably', 'involve', 'involves', 'involving', 'irrespective', 'itself', 'judged', 'kinds', 'lacks', 'largely', 'latter', 'lesser', 'likewise', 'limited', 'linked', 'mainly', 'majority', 'manner', 'matters', 'meaning', 'means', 'mere', 'merely', 'moreover', 'mostly', 'multitude', 'mutually', 'namely', 'naturally', 'necessarily', 'necessity', 'neither', 'neutral', 'nevertheless', 'nonetheless', 'nor', 'norm', 'normally', 'nowadays', 'obtainable', 'often', 'opposed', 'ordinarily', 'ordinary', 'originate', 'originating', 'otherwise', 'particular', 'particularly', 'partly', 'perceived', 'persons', 'pertain', 'pertaining', 'pertains', 'possess', 'possessing', 'possibility', 'preceding', 'precisely', 'predominantly', 'preferable', 'preference', 'preferences', 'preferred', 'presence', 'presently', 'presumably', 'prevailing', 'prevalent', 'primarily', 'principally', 'privilege', 'proportions', 'purely', 'purpose', 'purposes', 'rarely', 'readily', 'reasonable', 'reasonably', 'refer', 'referred', 'refers', 'regard', 'regarded', 'regardless', 'relation', 'relatively', 'relying', 'remain', 'remains', 'render', 'rendered', 'represent', 'represented', 'requires', 'requisite', 'resemble', 'reserved', 'reside', 'respects', 'restricted', 'satisfactory', 'satisfy', 'scenario', 'scheme', 'schemes', 'scope', 'signifies', 'signify', 'similar', 'similarly', 'simplest', 'situations', 'sole', 'solely', 'sorts', 'specific', 'specifically', 'strictly', 'strongly', 'subject', 'substantial', 'such', 'sufficient', 'suggesting', 'suggests', 'technically', 'tends', 'term', 'termed', 'terms', 'theoretically', 'therefore', 'thus', 'traditionally', 'transitional', 'typical', 'typically', 'unclear', 'uncommon', 'understood', 'universally', 'unknown', 'unlike', 'unlikely', 'unrelated', 'upon', 'usually', 'utilised', 'utilized', 'varied', 'various', 'versa', 'viewed', 'virtually', 'viz', 'whatsoever', 'whereas', 'whereby', 'whichever', 'wholly', 'widely']

SpeakingPop=['abba', 'acme', 'aesop', 'aladdin', 'allegro', 'andreas', 'andromeda', 'apollo', 'aquarius', 'aquila', 'ares', 'argo', 'aries', 'aristotle', 'arma', 'armada', 'armageddon', 'artemis', 'astro', 'athena', 'atlantis', 'beatles', 'beethoven', 'borg', 'brio', 'britannica', 'bros', 'caspian', 'cdm', 'cerberus', 'chernobyl', 'chiron', 'colossus', 'constantine', 'corby', 'csi', 'cso', 'custome', 'dante', 'darwin', 'diablo', 'dino', 'disney', 'dj', 'dlc', 'dna', 'dominator', 'dora', 'ecw', 'einstein', 'elan', 'enya', 'fahrenheit', 'fantasia', 'faq', 'faqs', 'fma', 'galileo', 'gamecube', 'gatsby', 'gba', 'gemini', 'genesis', 'giga', 'godfather', 'goliath', 'gotham', 'gta', 'hades', 'hazzard', 'hercules', 'hq', 'igi', 'ign', 'igo', 'ipa', 'iq', 'isis', 'isos', 'ita', 'izumi', 'janus', 'jpn', 'juno', 'jupiter', 'kahuna', 'kirby', 'launchpad', 'lego', 'legos', 'leo', 'leonardo', 'leone', 'lex', 'luigi', 'luna', 'lux', 'mario', 'maya', 'medusa', 'merlin', 'midas', 'millenium', 'mirage', 'morpheus', 'movi', 'mozart', 'napoleon', 'nds', 'neo', 'neptune', 'nes', 'nile', 'norse', 'nox', 'odyssey', 'omni', 'orion', 'ost', 'pascal', 'pegasus', 'pes', 'pikachu', 'pluto', 'potter', 'primus', 'prometheus', 'quicksilver', 'raider', 'rei', 'remi', 'remo', 'remy', 'rex', 'rhapsody', 'riva', 'roc', 'roland', 'romeo', 'sagittarius', 'salomon', 'sandman', 'saxon', 'scooby', 'screensaver', 'sega', 'sfx', 'shakespeare', 'shal', 'showtime', 'shrek', 'siberia', 'skywalker', 'socrates', 'sopranos', 'superman', 'tantra', 'tesla', 'thor', 'titan', 'titanic', 'tnt', 'torrent', 'trax', 'triad', 'twilight', 'umd', 'venus', 'vhs', 'viking', 'vinci', 'virgo', 'vod', 'voyager', 'vulcan', 'wwf', 'xtreme', 'yamato', 'zelda', 'zeus', 'ballerina', 'bam', 'barber', 'barbie', 'baron', 'bayou', 'bender', 'binky', 'bloomer', 'bong', 'bongo', 'bonny', 'booger', 'boomer', 'bop', 'broom', 'buccaneer', 'buddha', 'buster', 'busters', 'butch', 'butcher', 'butler', 'carpenter', 'carver', 'chap', 'cinderella', 'circus', 'claus', 'clown', 'coffin', 'cowboy', 'daisy', 'dame', 'dandy', 'daze', 'deuce', 'ditto', 'ditty', 'diva', 'dolly', 'domino', 'doo', 'doodle', 'dotty', 'elmo', 'exotica', 'fairy', 'fanny', 'ferris', 'fireman', 'foxtrot', 'genie', 'gent', 'gifs', 'goo', 'goodnight', 'grail', 'greeter', 'grinch', 'gypsy', 'hag', 'haircut', 'hooray', 'howdy', 'hula', 'jester', 'jingle', 'jive', 'jolly', 'jukebox', 'kleenex', 'lark', 'loo', 'madam', 'mascot', 'matchbox', 'merry', 'mickey', 'milliner', 'mime', 'minnie', 'mister', 'moe', 'mohawk', 'mojo', 'motley', 'mustache', 'natty', 'nomad', 'nutcracker', 'packer', 'paintbrush', 'papa', 'patsy', 'peep', 'pew', 'piggy', 'pilgrim', 'piper', 'pj', 'pokey', 'pom', 'pooh', 'pooper', 'porter', 'pow', 'prim', 'racecar', 'rag', 'rags', 'riddle', 'scrooge', 'shack', 'shakers', 'shoemaker', 'shoo', 'skippy', 'sleigh', 'smiley', 'snowball', 'snuff', 'soapbox', 'sparky', 'squire', 'stoner', 'storybook', 'swank', 'taffy', 'taker', 'teller', 'tiki', 'tinker', 'tumbleweed', 'tween', 'twister', 'usher', 'weaver', 'wee', 'weekender', 'willy', 'winnie', 'wolfman', 'wonderland', 'workman', 'yankee']

SpeakingGeek=['abs', 'ace', 'aero', 'ala', 'amazon', 'amp', 'ante', 'anti', 'arcade', 'atm', 'autos', 'betas', 'bio', 'biz', 'booster', 'boosters', 'bot', 'brainer', 'bucks', 'calculators', 'calender', 'cd', 'cds', 'cetera', 'cheat', 'cig', 'clone', 'coder', 'combo', 'commerical', 'comp', 'comps', 'cons', 'consoles', 'coolest', 'coop', 'crack', 'cred', 'demo', 'dent', 'digi', 'dime', 'dummies', 'dummy', 'dvds', 'ect', 'ergo', 'esp', 'essentials', 'etc', 'exclusives', 'extras', 'finders', 'fro', 'fx', 'gadget', 'gadgets', 'galore', 'geek', 'gen', 'gizmos', 'glitch', 'gov', 'hack', 'hacks', 'halo', 'haves', 'helper', 'hobbies', 'hobby', 'hookup', 'hyper', 'ie', 'indy', 'ins', 'junk', 'lastest', 'legit', 'lite', 'lowdown', 'macs', 'mechanic', 'mega', 'messenger', 'misc', 'mod', 'mods', 'nexus', 'nifty', 'non', 'nuke', 'offical', 'offs', 'ops', 'penny', 'perk', 'perks', 'phantom', 'playbook', 'pod', 'pong', 'pre', 'preps', 'pro', 'props', 'pros', 'psych', 'puzzle', 'quid', 'rad', 'rep', 'rip', 'saver', 'savers', 'schematics', 'sci', 'seperate', 'setups', 'shocker', 'shuffle', 'sig', 'sim', 'sims', 'skins', 'slash', 'spec', 'specs', 'splinter', 'strat', 'stub', 'subs', 'super', 'surefire', 'swap', 'tech', 'tele', 'tix', 'todays', 'ton', 'tracker', 'tri', 'tricks', 'uni', 'ups', 'vip', 'ware', 'wishlist', 'zap', 'zen']

SpeakingCasualAndFamily=['adoptive', 'age', 'aged', 'aunt', 'babysit', 'babysitter', 'babysitting', 'beloved', 'birth', 'born', 'boy', 'boyfriend', 'bride', 'brother', 'brothers', 'buddies', 'buddy', 'child', 'childhood', 'children', 'chores', 'classmate', 'classmates', 'companion', 'cousin', 'cousins', 'coworker', 'coworkers', 'dad', 'daddy', 'dads', 'darling', 'daughter', 'daughters', 'daycare', 'dear', 'dearest', 'deceased', 'divorced', 'elder', 'errands', 'exes', 'family', 'father', 'fathers', 'fianc', 'fiancee', 'friend', 'friends', 'girlfriend', 'godmother', 'godson', 'grandchild', 'grandchildren', 'grandfather', 'grandma', 'grandmother', 'grandpa', 'grandparents', 'grandson', 'groom', 'her', 'hubby', 'hugs', 'husband', 'husbands', 'kid', 'kids', 'kin', 'lad', 'loved', 'lover', 'loves', 'loving', 'mama', 'marriage', 'married', 'marry', 'mates', 'mom', 'mommy', 'mother', 'mothers', 'mum', 'mummy', 'nanny', 'neighbor', 'neighbors', 'neighbour', 'neighbours', 'nephew', 'niece', 'nieces', 'offspring', 'old', 'older', 'orphan', 'orphanage', 'orphaned', 'pal', 'pals', 'parent', 'parents', 'pregnant', 'relatives', 'roommate', 'roommates', 'sibling', 'siblings', 'sister', 'sisters', 'sitter', 'son', 'sons', 'spouse', 'surrogate', 'survivor', 'sweetheart', 'teenager', 'twins', 'uncle', 'uncles', 'wed', 'weds', 'widow', 'wife', 'younger', 'youngest', 'yr', 'yrs']


SpeakingMachismo=['adjustable', 'adjuster', 'aftermarket', 'anchors', 'arbor', 'arm', 'assembly', 'attach', 'attaches', 'attachment', 'attachments', 'axel', 'bearing', 'bearings', 'blade', 'blades', 'bolt', 'bolted', 'bolts', 'brace', 'bracket', 'brackets', 'brake', 'brakes', 'bucket', 'bumper', 'bumpers', 'cage', 'carriage', 'chain', 'chains', 'chassis', 'chrome', 'chute', 'clamp', 'clutch', 'cone', 'cradle', 'crane', 'crank', 'crawler', 'dash', 'disc', 'drill', 'extenders', 'fins', 'flatbed', 'flex', 'flush', 'flywheel', 'folding', 'forged', 'fork', 'forks', 'friction', 'gear', 'gearbox', 'gearing', 'gears', 'glide', 'grille', 'grip', 'grips', 'handlebar', 'hanger', 'harness', 'hatch', 'hauler', 'headlights', 'hex', 'hinge', 'hinges', 'hitch', 'hook', 'hooks', 'hub', 'hubs', 'jacks', 'jaws', 'keyed', 'kit', 'knob', 'ladder', 'lever', 'lift', 'linkage', 'lock', 'locking', 'locks', 'mast', 'mated', 'mirror', 'mount', 'mounted', 'mounting', 'mounts', 'mower', 'parts', 'pedal', 'peg', 'pickup', 'pickups', 'pin', 'pinion', 'pins', 'pivot', 'plate', 'plow', 'pole', 'poles', 'prop', 'rack', 'racks', 'rails', 'rake', 'ratchet', 'ratcheting', 'rear', 'reel', 'restraints', 'retainer', 'retainers', 'retract', 'retractable', 'rim', 'rims', 'riser', 'rocker', 'rod', 'rods', 'roller', 'rotate', 'rusted', 'saddle', 'sander', 'screw', 'screws', 'seat', 'seats', 'shaft', 'shank', 'shim', 'shock', 'shocks', 'skid', 'slant', 'sliding', 'spare', 'spinner', 'spool', 'springs', 'stabilizer', 'stabilizers', 'steering', 'stopper', 'suspension', 'sway', 'tail', 'tailgate', 'tarp', 'tighten', 'tiller', 'tilt', 'tire', 'tires', 'tow', 'traction', 'tractor', 'tread', 'trimmer', 'tripod', 'trunk', 'truss', 'wagon', 'wheel', 'wheeled', 'wheels', 'windshield', 'wrench', 'acura', 'alfa', 'astra', 'audi', 'awd', 'bentley', 'benz', 'bmw', 'bonneville', 'boxster', 'bronco', 'buick', 'cadillac', 'carlo', 'carrera', 'cavalier', 'cdi', 'cherokee', 'chevrolet', 'chevy', 'chrysler', 'cj', 'cng', 'cobra', 'convertible', 'convertibles', 'corolla', 'corvette', 'cummins', 'cutlass', 'daewoo', 'daimler', 'davidson', 'deere', 'deville', 'dodge', 'dsm', 'dunlop', 'durango', 'ecm', 'edmunds', 'efi', 'engin', 'escalade', 'ferrari', 'fiat', 'ford', 'forester', 'galant', 'gmc', 'goodyear', 'gsi', 'gsx', 'gts', 'harley', 'haynes', 'headlight', 'highlander', 'holden', 'honda', 'hummer', 'hyundai', 'innova', 'isuzu', 'jaguar', 'jeep', 'kawasaki', 'lancer', 'lexus', 'lpg', 'lx', 'maf', 'malibu', 'maruti', 'mclaren', 'mercedes', 'mgb', 'miata', 'mitsubishi', 'mustang', 'mustangs', 'nissan', 'oem', 'oldsmobile', 'pacifica', 'pirelli', 'polaris', 'pontiac', 'porsche', 'prowler', 'rav', 'redline', 'renault', 'rhd', 'roadrunner', 'rover', 'royce', 'saab', 'saturn', 'scrambler', 'sedan', 'sema', 'shelby', 'sierra', 'silverado', 'stang', 'sti', 'studebaker', 'suv', 'suvs', 'suzuki', 'tahoe', 'talon', 'taurus', 'tbi', 'tdi', 'thunderbird', 'tj', 'toyota', 'trailblazer', 'trd', 'triton', 'turbo', 'ute', 'vin', 'volvo', 'vw', 'wrc', 'xb', 'xj', 'yj', 'yukon', 'aces', 'against', 'battled', 'beat', 'beating', 'berth', 'birdie', 'bogey', 'bogeys', 'breakaway', 'champ', 'champion', 'champions', 'championship', 'championships', 'champs', 'clash', 'clinch', 'clinching', 'comeback', 'compete', 'competed', 'competes', 'competing', 'competition', 'competitor', 'competitors', 'consecutive', 'consolation', 'contender', 'contenders', 'contest', 'contestant', 'contestants', 'contested', 'contests', 'decisive', 'defeat', 'defeated', 'defeating', 'defeats', 'defending', 'derby', 'discus', 'division', 'divisional', 'dominated', 'dominating', 'doubles', 'earned', 'eighth', 'eventual', 'fielded', 'fifth', 'final', 'finalist', 'finalists', 'finals', 'finishes', 'finishing', 'fourth', 'handicap', 'handily', 'heavyweight', 'hurdle', 'invitational', 'javelin', 'lone', 'longshot', 'losing', 'match', 'matches', 'matchup', 'matchups', 'medal', 'medals', 'narrowly', 'ninth', 'opener', 'opponent', 'opponents', 'overcame', 'pairings', 'pennant', 'penultimate', 'pitted', 'playoff', 'podium', 'postseason', 'putt', 'qualifier', 'qualifiers', 'qualifying', 'race', 'races', 'ranked', 'ranking', 'rankings', 'reigning', 'rival', 'rivalry', 'rivals', 'rout', 'runner', 'runners', 'score', 'scores', 'seeded', 'semifinals', 'seventh', 'shootout', 'showdown', 'singles', 'sixth', 'slam', 'sprint', 'sprinters', 'standings', 'streak', 'tally', 'tiebreaker', 'title', 'toughest', 'tournament', 'tourney', 'triumph', 'trophy', 'unbeaten', 'undefeated', 'underdog', 'upsets', 'victories', 'victory', 'vs', 'win', 'winner', 'winners', 'winning', 'wins', 'won']

SpeakingHumanity=['amazement', 'angrily', 'answered', 'anxiously', 'asked', 'asleep', 'awoke', 'begged', 'beside', 'blinked', 'bowed', 'breathed', 'calmed', 'calmly', 'casually', 'chatted', 'chatting', 'cheered', 'cried', 'crowed', 'curiously', 'curled', 'danced', 'dawned', 'eagerly', 'exclaimed', 'exclaims', 'eyebrows', 'eyed', 'eyeing', 'fainted', 'fondly', 'frantically', 'frown', 'frowned', 'gaze', 'glance', 'glanced', 'graciously', 'grasped', 'gratefully', 'greeted', 'grin', 'grumbled', 'happily', 'hastily', 'hers', 'hesitated', 'huddled', 'hug', 'hugged', 'hurried', 'hurriedly', 'inquired', 'instinctively', 'intently', 'interrupted', 'involuntarily', 'joked', 'kissed', 'knelt', 'laughed', 'leaned', 'listened', 'looked', 'momentarily', 'nervously', 'nod', 'nodded', 'noticing', 'overheard', 'parted', 'patiently', 'paused', 'peering', 'phoned', 'politely', 'pondered', 'quietly', 'quipped', 'rapt', 'reluctantly', 'remarked', 'replied', 'rested', 'sat', 'screamed', 'seated', 'sensed', 'shivering', 'shook', 'shouted', 'shrug', 'shrugged', 'shrugging', 'sigh', 'silently', 'slept', 'smelled', 'smile', 'smiled', 'smiles', 'smiling', 'snarled', 'softly', 'stare', 'stared', 'staring', 'startled', 'stood', 'straightened', 'suddenly', 'suspiciously', 'swayed', 'swore', 'teased', 'thanked', 'trembling', 'waited', 'walked', 'wandered', 'watched', 'waved', 'waving', 'whispered', 'woke', 'wondered', 'yelled', 'about', 'actually', 'admit', 'afraid', 'again', 'agree', 'almost', 'alone', 'already', 'although', 'always', 'am', 'anybody', 'anymore', 'anyone', 'anything', 'anyway', 'aside', 'asking', 'assume', 'assuming', 'awhile', 'bad', 'barely', 'basically', 'because', 'being', 'believe', 'besides', 'bit', 'blame', 'bother', 'bothered', 'busy', 'but', 'ca', 'cared', 'certainly', 'clue', 'coming', 'completely', 'considering', 'convinced', 'could', 'couple', 'curious', 'decent', 'definately', 'definitely', 'deserve', 'deserved', 'deserves', 'did', 'didn', 'does', 'doing', 'done', 'doubt', 'else', 'enough', 'especially', 'even', 'ever', 'everybody', 'everyone', 'everything', 'everywhere', 'exactly', 'excited', 'expecting', 'explain', 'fact', 'familiar', 'far', 'fault', 'feel', 'feeling', 'feels', 'felt', 'few', 'figure', 'figured', 'figuring', 'folks', 'forget', 'forgetting', 'forgot', 'fortunately', 'getting', 'glad', 'going', 'gone', 'good', 'got', 'gotten', 'guess', 'guessed', 'guessing', 'guys', 'happen', 'happened', 'happening', 'happens', 'happy', 'hard', 'hardly', 'he', 'hear', 'heard', 'heck', 'him', 'honestly', 'hope', 'hopefully', 'hoping', 'how', 'hurt', 'idea', 'ignore', 'imagine', 'impossible', 'impression', 'indeed', 'just', 'kind', 'knew', 'know', 'knowing', 'knows', 'lately', 'learned', 'letting', 'like', 'liked', 'likes', 'liking', 'literally', 'little', 'looks', 'lost', 'lot', 'love', 'luck', 'luckily', 'lucky', 'man', 'matter', 'mattered', 'maybe', 'me', 'mean', 'meant', 'mention', 'mentioned', 'might', 'mind', 'mine', 'miss', 'missed', 'missing', 'mistake', 'moment', 'much', 'my', 'myself', 'never', 'nobody', 'none', 'nothing', 'noticed', 'now', 'nowhere', 'obvious', 'obviously', 'once', 'ours', 'perhaps', 'personally', 'possibly', 'practically', 'pretty', 'probably', 'problem', 'putting', 'quit', 'quite', 'quitting', 'rather', 'realise', 'realised', 'realize', 'realized', 'realizing', 'really', 'reason', 'recall', 'reckon', 'remember', 'remind', 'reminded', 'sadly', 'say', 'saying', 'seeing', 'seem', 'seemed', 'seems', 'seen', 'serious', 'seriously', 'she', 'situation', 'so', 'solved', 'somebody', 'someday', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhere', 'soon', 'sooner', 'sort', 'staying', 'still', 'stuff', 'suppose', 'supposed', 'sure', 'surely', 'surprise', 'surprised', 'talk', 'talking', 'tell', 'telling', 'thank', 'thankful', 'thankfully', 'that', 'theirs', 'there', 'thing', 'things', 'think', 'thinking', 'though', 'thought', 'too', 'totally', 'tried', 'trouble', 'trying', 'understand', 'unfortunately', 'wait', 'waiting', 'wanted', 'wanting', 'watching', 'what', 'whatever', 'when', 'whoever', 'whole', 'why', 'wo', 'wonder', 'wondering', 'worry', 'worrying', 'worse', 'would', 'wrong', 'yes', 'yet', 'yours']

SpeakingDramatic=['abuzz', 'advent', 'afoot', 'airwaves', 'amid', 'anticipating', 'anticipation', 'awaited', 'awash', 'backlash', 'ballyhooed', 'bandwagon', 'barrage', 'binge', 'blip', 'blitz', 'blowout', 'bode', 'bodes', 'bonanza', 'boom', 'booming', 'breakup', 'buzz', 'clamoring', 'coincided', 'coincides', 'controversy', 'courting', 'craze', 'craziness', 'culminated', 'culminating', 'dearth', 'debacle', 'deluge', 'demise', 'doomsday', 'dotcom', 'downturn', 'droves', 'emergence', 'ensued', 'ensuing', 'euphoria', 'exodus', 'fad', 'fallout', 'fanfare', 'feud', 'fiasco', 'firestorm', 'fizzled', 'flurry', 'frenzy', 'fueled', 'fueling', 'fuelled', 'fuelling', 'furor', 'glut', 'groundswell', 'harbinger', 'headlines', 'hinting', 'hype', 'hyped', 'ignited', 'imminent', 'impetus', 'implosion', 'influx', 'intensified', 'intensifying', 'inundated', 'jitters', 'jolt', 'juggernaut', 'knell', 'leaked', 'looms', 'lull', 'mania', 'massive', 'meltdown', 'meteoric', 'momentum', 'mushrooming', 'nascent', 'nearing', 'outcry', 'overshadowed', 'overtaken', 'phenomenon', 'prelude', 'prompted', 'prompting', 'propelled', 'publicity', 'publicized', 'rallying', 'ramping', 'raves', 'recession', 'resurgence', 'revival', 'rift', 'rumblings', 'rumor', 'rumors', 'rumour', 'rumours', 'runaway', 'runup', 'salvo', 'sellout', 'setback', 'shakeout', 'shakeup', 'shortage', 'signaled', 'slump', 'spark', 'sparked', 'sparking', 'spate', 'spawned', 'speculating', 'speculation', 'spike', 'spillover', 'spur', 'spurred', 'spurring', 'spurs', 'spurt', 'stampede', 'stemmed', 'stoked', 'storming', 'surfaced', 'surge', 'transpired', 'trend', 'underway', 'uproar', 'upswing', 'uptick', 'ushered', 'waning', 'whammy', 'whirlwind', 'accuse', 'accusing', 'adamant', 'adamantly', 'admitting', 'advocated', 'angering', 'apologized', 'apologizing', 'applaud', 'applauded', 'assailed', 'backpedaled', 'balked', 'bashed', 'belatedly', 'betrayed', 'bitterly', 'blamed', 'blaming', 'blithely', 'bluntly', 'campaigned', 'candidly', 'cautioned', 'challenged', 'championed', 'chastised', 'chided', 'commended', 'complained', 'conceded', 'conceding', 'concurred', 'condemn', 'condemned', 'condemning', 'confessed', 'congratulated', 'congratulating', 'cooperated', 'counseled', 'countered', 'courted', 'criticised', 'criticize', 'criticized', 'criticizing', 'critics', 'cussed', 'decried', 'defended', 'defied', 'defying', 'denounced', 'derided', 'deterred', 'detractors', 'disagreed', 'disapproved', 'disrespected', 'doubted', 'downplayed', 'embraced', 'emphatically', 'endorsing', 'excused', 'faulted', 'fended', 'flatly', 'foolishly', 'forgave', 'harassed', 'harshly', 'hinted', 'hounded', 'ignored', 'implored', 'insisted', 'insistence', 'insisting', 'insulted', 'intervened', 'justifiably', 'lambasted', 'lamented', 'lauded', 'lectured', 'likened', 'lobbied', 'maligned', 'mocked', 'naysayers', 'objected', 'objecting', 'openly', 'opined', 'overtures', 'persuaded', 'pilloried', 'pointedly', 'praised', 'praising', 'pressured', 'pressuring', 'protested', 'publically', 'publicly', 'qualms', 'questioned', 'raved', 'reacted', 'rebuffed', 'refrained', 'refusing', 'reiterating', 'reneged', 'repeatedly', 'resisted', 'resorted', 'rightfully', 'rightly', 'risked', 'roundly', 'scapegoat', 'scoffed', 'shunned', 'singling', 'speculated', 'spokesmen', 'staunch', 'steadfastly', 'stonewalled', 'strenuously', 'trumped', 'trumpeted', 'undermined', 'unjustly', 'unsuccessfully', 'vehemently', 'verbally', 'vilified', 'vindicated', 'voiced', 'voicing', 'vowed', 'vowing', 'warned']

SpeakingBanter=['ai', 'ain', 'aint', 'aye', 'ballers', 'befo', 'bein', 'bout', 'boyz', 'bub', 'buildin', 'buyin', 'cali', 'cha', 'changin', 'checkin', 'comin', 'coo', 'cookin', 'crankin', 'cruisin', 'da', 'dank', 'darlin', 'dat', 'dawg', 'dawgs', 'dey', 'diddy', 'dimes', 'dis', 'doin', 'dolla', 'dope', 'dreamin', 'dro', 'dubs', 'dum', 'em', 'everythin', 'fam', 'fellas', 'fer', 'fightin', 'fink', 'fishin', 'flo', 'fo', 'followin', 'gat', 'gats', 'gettin', 'gimme', 'goin', 'gon', 'graff', 'havin', 'hearin', 'holdin', 'ima', 'iz', 'jammin', 'jes', 'joc', 'keepin', 'lil', 'locs', 'lookin', 'lov', 'lovin', 'mah', 'makin', 'mane', 'meetin', 'mek', 'missin', 'mofo', 'momma', 'mornin', 'movin', 'nas', 'nig', 'nothin', 'nuff', 'ole', 'outta', 'pac', 'pimp', 'pimpin', 'poppa', 'purdy', 'rapper', 'rhymes', 'rockin', 'rollin', 'runnin', 'savin', 'sayin', 'sellin', 'sez', 'sho', 'shorty', 'spendin', 'tch', 'tha', 'thang', 'tonite', 'tru', 'tryin', 'tupac', 'usin', 'wack', 'waitin', 'wassup', 'westcoast', 'wha', 'whassup', 'wid', 'wit', 'wiz', 'workin', 'wuz', 'yall', 'yer', 'yo', 'yoo', 'afterall', 'ah', 'aha', 'alot', 'alright', 'anyhow', 'anyways', 'arent', 'asap', 'atleast', 'aw', 'awsome', 'bday', 'bf', 'boo', 'bro', 'btw', 'bummer', 'bye', 'cant', 'congrats', 'cos', 'couldnt', 'cus', 'cuz', 'dammit', 'damn', 'dang', 'didnt', 'doesnt', 'dont', 'dude', 'duh', 'dunno', 'eek', 'eh', 'erm', 'everytime', 'fav', 'freakin', 'ftw', 'fyi', 'gee', 'geez', 'gf', 'gonna', 'gosh', 'gotta', 'ha', 'haha', 'hav', 'havent', 'hee', 'heh', 'hello', 'heres', 'hes', 'hey', 'hi', 'hmm', 'hmmm', 'hmmmm', 'hoo', 'hows', 'huh', 'hun', 'i', 'iam', 'im', 'imo', 'ish', 'isnt', 'ive', 'jeez', 'jk', 'jus', 'kidding', 'kinda', 'kno', 'kool', 'lemme', 'lol', 'luv', 'mate', 'meh', 'mmm', 'nah', 'nevermind', 'nope', 'oh', 'ok', 'okay', 'ooo', 'oooo', 'oops', 'ouch', 'peeps', 'pls', 'plz', 'ppl', 'prob', 'realy', 'rly', 'shud', 'sir', 'sis', 'soo', 'sooo', 'soooo', 'sooooo', 'sorry', 'thankyou', 'thanx', 'thats', 'theres', 'thnx', 'thx', 'u', 'ugh', 'uh', 'ur', 'vid', 'wanna', 'wasnt', 'wat', 'wen', 'whats', 'whew', 'whoa', 'whos', 'wierd', 'wink', 'wont', 'woo', 'woohoo', 'wouldnt', 'wow', 'wud', 'ya', 'yah', 'yea', 'yeah', 'yeh', 'yep', 'yikes', 'youre', 'yup', 'addicted', 'assholes', 'bashing', 'bastard', 'bastards', 'bitching', 'blah', 'bored', 'bothering', 'bothers', 'bragging', 'brains', 'brat', 'bugging', 'bullshit', 'bully', 'bum', 'bums', 'bunch', 'cares', 'complain', 'complaining', 'crap', 'crappy', 'crazy', 'damned', 'darn', 'disgusting', 'dislike', 'drool', 'drunk', 'drunken', 'dudes', 'dumb', 'dumbest', 'excuse', 'excuses', 'fag', 'fake', 'fart', 'farts', 'fella', 'fool', 'fools', 'frat', 'freak', 'freaking', 'freaks', 'freaky', 'friggin', 'fuckin', 'funny', 'garbage', 'gripe', 'guts', 'guy', 'hate', 'hated', 'hates', 'hell', 'horrible', 'idiot', 'idiots', 'ignorant', 'insane', 'insult', 'insulting', 'insults', 'jackass', 'jealous', 'jerk', 'joke', 'jokes', 'joking', 'lame', 'laugh', 'laughing', 'lazy', 'liar', 'loser', 'losers', 'lying', 'mad', 'meme', 'mess', 'messed', 'messes', 'messing', 'moron', 'morons', 'nerd', 'nonsense', 'obsessed', 'pathetic', 'picky', 'piss', 'pissed', 'poo', 'poop', 'pooping', 'prank', 'pretend', 'pretending', 'pricks', 'psycho', 'puke', 'racist', 'ranting', 'raving', 'redneck', 'retard', 'retarded', 'rotten', 'rude', 'ruined', 'ruining', 'scare', 'scares', 'scaring', 'screwed', 'shit', 'shitty', 'shove', 'sick', 'slacker', 'slap', 'smack', 'smelly', 'sniff', 'snot', 'spew', 'spewing', 'spit', 'spoiled', 'stereotype', 'stink', 'stinking', 'stinks', 'stinky', 'stupid', 'stupidest', 'swear', 'thier', 'tired', 'trash', 'turd', 'ugly', 'wannabe', 'wasted', 'wasting', 'whack', 'whining', 'worst', 'worthless', 'yanks', 'yuck']

SpeakingProfanity=['Arse',
'Bloody',
'Bugger',
'Cow',
'Crap',
'Damn',
'Ginger',
'Git',
'God',
'Goddam',
'Jesus',
'Christ',
'Minger',
'Sod',
'Arsehole',
'Balls',
'Bint',
'Bitch',
'Bollocks',
'Bullshit',
'Feck',
'Munter',
'Pissed',
'Shit',
'Tits ',
'Bastard',
'Beaver',
'Bellend',
'Bloodclaat',
'Clunge',
'Cock',
'Dick',
'Dickhead',
'Fanny',
'Flaps',
'Gash',
'Knob',
'Minge',
'Prick',
'Punani',
'Pussy',
'Snatch',
'Twat ',
'Strongest',
'Cunt',
'Fuck',
'Motherfucker',
'ass']

#ACTING

ActingUrgency=['abandon', 'abreast', 'abstain', 'accompany', 'accustomed', 'acknowledge', 'acquaintance', 'acquaintances', 'acquainted', 'adage', 'admonition', 'advisable', 'alike', 'allude', 'amaze', 'amuse', 'anticipate', 'apprise', 'apprised', 'apt', 'aren', 'ascertain', 'aspire', 'attest', 'avail', 'await', 'balk', 'bargained', 'barter', 'beforehand', 'beg', 'behave', 'beware', 'blindly', 'borrow', 'brag', 'budge', 'caution', 'cease', 'cede', 'cherish', 'cling', 'cognizant', 'comers', 'commend', 'compelled', 'comprehend', 'conceivably', 'confess', 'confide', 'contemplate', 'contend', 'conversant', 'converse', 'conversely', 'convince', 'cooperate', 'cordially', 'couldn', 'cram', 'crapper', 'crave', 'dare', 'dedicate', 'deem', 'defy', 'delve', 'desiring', 'deviate', 'devote', 'dictate', 'disappoint', 'disapprove', 'discern', 'dispense', 'divulge', 'doesn', 'don', 'embark', 'endeavoring', 'endeavour', 'enlighten', 'enlist', 'entertain', 'entice', 'enticed', 'entrust', 'envisage', 'envision', 'err', 'eventuality', 'existent', 'expedient', 'fathom', 'firstly', 'foolproof', 'forbear', 'forbid', 'forego', 'foresee', 'forewarned', 'forgo', 'fret', 'fulfil', 'fuss', 'gladly', 'gobble', 'goings', 'grasp', 'gravitate', 'haggle', 'heed', 'hesitate', 'hesitation', 'hoard', 'hunch', 'hunker', 'hurry', 'impress', 'inclination', 'inclined', 'indulge', 'indulging', 'inquire', 'insist', 'instruct', 'intend', 'intending', 'iota', 'jot', 'juncture', 'laboring', 'lastly', 'leeway', 'lend', 'lest', 'loath', 'masse', 'materialize', 'meantime', 'mingle', 'mull', 'muster', 'necessitate', 'necessities', 'nigh', 'nudge', 'obey', 'oblige', 'obliged', 'observe', 'offing', 'oft', 'oftentimes', 'opine', 'opportune', 'opting', 'ought', 'pardon', 'perceive', 'persuade', 'peruse', 'ponder', 'pondering', 'predisposed', 'preferring', 'presume', 'prevail', 'privy', 'proceed', 'procure', 'prosper', 'react', 'reap', 'reassure', 'reciprocate', 'recognise', 'refrain', 'refuse', 'regress', 'regrettably', 'relent', 'relinquish', 'remiss', 'resent', 'resist', 'retire', 'revolve', 'savor', 'scarcely', 'scrutinize', 'secondly', 'seize', 'seldom', 'shun', 'slightest', 'someplace', 'speculate', 'spoil', 'steer', 'stumble', 'succumb', 'suffice', 'sympathize', 'tarry', 'tempted', 'tending', 'thirdly', 'tolerate', 'underestimate', 'undivided', 'uninitiated', 'unquestionably', 'unwilling', 'upshot', 'urge', 'veer', 'venturing', 'vouch', 'warn', 'westerners', 'wherewithal', 'whet', 'whim', 'wholeheartedly', 'whomever', 'willingly', 'wished', 'wishes', 'wishing', 'wrestle', 'yourselves', 'abruptly', 'accidentally', 'accidently', 'banked', 'battered', 'beaten', 'blacked', 'blasted', 'bled', 'blew', 'blown', 'booted', 'bounced', 'braced', 'broke', 'bucked', 'bumped', 'burned', 'chased', 'choked', 'circled', 'climbed', 'clipped', 'clobbered', 'clocked', 'collapsed', 'cracked', 'cranked', 'crashed', 'crawled', 'crept', 'crossed', 'crushed', 'dashed', 'dented', 'dodged', 'downed', 'dragged', 'drifted', 'dropped', 'drove', 'drowned', 'dug', 'dumped', 'eased', 'edged', 'electrocuted', 'exited', 'exploded', 'fallen', 'fell', 'flashed', 'flattened', 'flipped', 'floated', 'flushed', 'froze', 'gobbled', 'gouged', 'grabbed', 'hammered', 'hauled', 'hid', 'hosed', 'hovered', 'hung', 'inched', 'jacked', 'jammed', 'jolted', 'jumped', 'kicked', 'knocked', 'landed', 'lashed', 'layed', 'leaped', 'leapt', 'leveled', 'lifted', 'loosened', 'lowered', 'mysteriously', 'narrowed', 'neared', 'picked', 'piled', 'pinned', 'piped', 'plunged', 'pocketed', 'popped', 'poured', 'pressed', 'probed', 'prodded', 'propped', 'pulled', 'pummeled', 'pumped', 'punched', 'pushed', 'raced', 'racked', 'raked', 'rallied', 'ran', 'rattled', 'regained', 'retreated', 'ridden', 'ripped', 'roared', 'rocked', 'rode', 'rolled', 'roped', 'rung', 'rushed', 'sank', 'scooped', 'scoured', 'scrambled', 'scraped', 'scratched', 'scrubbed', 'shattered', 'shifted', 'skimmed', 'skipped', 'slammed', 'slapped', 'slashed', 'slid', 'slipped', 'slumped', 'smacked', 'smashed', 'smoothed', 'snapped', 'socked', 'spat', 'spiked', 'spilled', 'sprang', 'spun', 'squeezed', 'staggered', 'staked', 'stepped', 'strapped', 'strayed', 'stretched', 'struck', 'struggled', 'stumbled', 'stung', 'swallowed', 'swept', 'swerved', 'swung', 'tacked', 'tackled', 'tapped', 'threw', 'thrown', 'ticked', 'tightened', 'tilted', 'tipped', 'tore', 'tossed', 'touched', 'trailed', 'tripped', 'tumbled', 'unexpectedly', 'vanished', 'warmed', 'weighed', 'whacked', 'widened', 'wiped', 'wrecked', 'yanked', 'zoomed']

ActingIndustryJargon=['abacus', 'amble', 'aqueduct', 'bagger', 'barb', 'barbells', 'barbs', 'barrow', 'batten', 'berm', 'boatman', 'bogie', 'boma', 'bund', 'burley', 'burr', 'bushman', 'buss', 'buttress', 'caboose', 'candlesticks', 'carmine', 'cartwheel', 'chads', 'christen', 'cilia', 'cinnabar', 'clapper', 'clipper', 'cobble', 'coffer', 'combs', 'coned', 'conger', 'conning', 'coops', 'cornfield', 'corral', 'cotten', 'croaker', 'crouch', 'cush', 'cutty', 'dais', 'dater', 'deadhead', 'dees', 'dimpled', 'docker', 'doormats', 'duns', 'enders', 'feets', 'fetter', 'fieldhouse', 'fives', 'flagpole', 'fletch', 'floater', 'fob', 'fourths', 'fulcrum', 'furrow', 'gallop', 'gill', 'gimble', 'gob', 'gouge', 'hammerheads', 'handspring', 'hardtail', 'hark', 'helms', 'hew', 'hock', 'huntsman', 'itin', 'jammer', 'jimmies', 'kerb', 'knop', 'laterals', 'lath', 'lingual', 'lino', 'lipped', 'lob', 'mandola', 'masthead', 'matts', 'maw', 'mazy', 'mids', 'millstone', 'moai', 'moire', 'mops', 'mullets', 'muzzy', 'nacelle', 'nickles', 'nipper', 'nits', 'nob', 'nock', 'norther', 'nosed', 'obi', 'ollie', 'pacer', 'panos', 'papered', 'parkers', 'pars', 'peck', 'petri', 'pikes', 'plough', 'plumb', 'poon', 'prods', 'pronged', 'putti', 'quill', 'rafter', 'raker', 'ratchets', 'rawhide', 'redstone', 'reedy', 'ricochet', 'rook', 'roper', 'ruff', 'sag', 'scrimshaw', 'sear', 'shanks', 'shaper', 'shears', 'shute', 'silvers', 'slates', 'slinger', 'slink', 'spar', 'spinnaker', 'stamper', 'stiles', 'stinger', 'stocker', 'stockman', 'stoops', 'stow', 'straddle', 'streamer', 'stringer', 'stumpy', 'tailer', 'teeter', 'telescoped', 'thrasher', 'tipper', 'topside', 'trapeze', 'trebuchet', 'trident', 'trued', 'tufts', 'tyer', 'unconstructed', 'unlit', 'unwound', 'vermilion', 'wacker', 'weir', 'wester', 'wheelhouse', 'whited', 'withers', 'yorkers', 'zig', 'adapted', 'adhered', 'adjacent', 'affixed', 'aligned', 'aligning', 'alignment', 'alternating', 'anchoring', 'apex', 'arranged', 'arrangement', 'articulating', 'attached', 'attaching', 'axis', 'backbone', 'bifurcated', 'bonding', 'bridged', 'bridging', 'circular', 'coiled', 'complementary', 'comprise', 'comprises', 'comprising', 'concentric', 'conductor', 'conforming', 'connecting', 'constituting', 'converging', 'conveying', 'core', 'corresponding', 'coupled', 'defining', 'dimension', 'disposed', 'diverging', 'dividing', 'downstream', 'element', 'elements', 'enclose', 'enclosing', 'enlarged', 'extending', 'externally', 'facet', 'formed', 'forming', 'grid', 'grids', 'groove', 'grooves', 'grounding', 'hollow', 'horizontal', 'imbedded', 'inner', 'inserted', 'inserting', 'insertion', 'integral', 'interconnect', 'interconnected', 'interconnecting', 'interconnection', 'interconnections', 'interconnects', 'interlocking', 'intermediate', 'invention', 'inverted', 'inward', 'lateral', 'layer', 'layers', 'mechanically', 'notches', 'openings', 'orientation', 'oriented', 'outer', 'outward', 'overlap', 'overlapped', 'overlapping', 'overlaps', 'parallel', 'passage', 'perimeter', 'peripheral', 'planar', 'portion', 'portions', 'positioned', 'positioning', 'predetermined', 'preferably', 'prism', 'projecting', 'projection', 'proximate', 'recess', 'rectangular', 'reflective', 'reinforcing', 'resilient', 'respective', 'retaining', 'retracted', 'retraction', 'rigid', 'rotated', 'rotating', 'rows', 'sections', 'segment', 'segmented', 'segments', 'selectively', 'separated', 'separating', 'separation', 'sequentially', 'spaced', 'spacer', 'spacing', 'splice', 'stationary', 'strands', 'structure', 'structures', 'substantially', 'surface', 'surfaces', 'symmetrical', 'tangential', 'terminating', 'therefrom', 'therein', 'thereon', 'trenches', 'triangular', 'upstream', 'vertical', 'vertically', 'wafer', 'wherein']

ActingOfficialeseAndLegalese=['acknowledging', 'affirm', 'affirmation', 'affirmative', 'affirming', 'agnostic', 'anecdotal', 'arguable', 'argue', 'argues', 'arguing', 'argument', 'arguments', 'ascribe', 'assert', 'asserted', 'asserting', 'assertion', 'assertions', 'asserts', 'assumption', 'attribution', 'basing', 'belief', 'believing', 'bias', 'categorically', 'caveat', 'certainty', 'cite', 'claim', 'cogent', 'concede', 'conclude', 'concluding', 'conclusion', 'conclusions', 'conclusive', 'concur', 'conjecture', 'consensus', 'contending', 'contention', 'contradict', 'contradicted', 'contradiction', 'contradicts', 'contrary', 'convincing', 'credence', 'credible', 'crux', 'culpability', 'damning', 'debatable', 'declarations', 'defensible', 'definitively', 'denial', 'denials', 'deny', 'denying', 'disagree', 'disagreeing', 'disagreement', 'dismiss', 'dismissing', 'disputing', 'disregard', 'disregarding', 'doubtful', 'equate', 'equated', 'evidence', 'explicitly', 'eyewitness', 'facie', 'facts', 'factual', 'factually', 'favoring', 'foregone', 'germane', 'hindsight', 'hypothetical', 'ignoring', 'immaterial', 'implication', 'implicitly', 'implied', 'imply', 'implying', 'impossibility', 'inasmuch', 'inconclusive', 'infer', 'inference', 'insofar', 'intents', 'invalidate', 'invoking', 'judgement', 'judgements', 'judgments', 'justifiable', 'justification', 'justifications', 'justified', 'justifies', 'justify', 'justifying', 'legitimacy', 'legitimate', 'litmus', 'logically', 'maxim', 'merit', 'merits', 'moot', 'motive', 'motives', 'negated', 'notwithstanding', 'objection', 'objections', 'objectively', 'onus', 'outset', 'parentage', 'particularity', 'persuasive', 'plainly', 'plausible', 'posit', 'precedent', 'predicated', 'premise', 'presumed', 'presumes', 'presuming', 'presumption', 'probable', 'pronouncements', 'proof', 'proponents', 'proposition', 'propositions', 'prove', 'proving', 'purport', 'purported', 'purports', 'questioning', 'rationale', 'rationalization', 'reasoned', 'reasoning', 'rebut', 'rebutting', 'reconciled', 'refute', 'refuted', 'reject', 'rejecting', 'reliance', 'relied', 'repudiation', 'restate', 'skeptics', 'speculative', 'statements', 'strawman', 'substantiate', 'substantiated', 'substantiation', 'substantive', 'supposition', 'surmise', 'tacit', 'tantamount', 'tenet', 'twofold', 'unambiguous', 'unambiguously', 'unequivocally', 'validity', 'veracity', 'verifiable', 'vindication', 'warranted', 'abide', 'accordance', 'accorded', 'affirmatively', 'aforesaid', 'alia', 'annexed', 'arising', 'ascertained', 'authorise', 'authorised', 'authorize', 'authorizes', 'authorizing', 'bona', 'breach', 'breached', 'complied', 'conditionally', 'confer', 'conferred', 'conferring', 'confidentiality', 'consent', 'consented', 'consenting', 'consents', 'consequential', 'constitute', 'constituted', 'construed', 'consummated', 'contemplated', 'contemporaneously', 'continuance', 'contractual', 'contractually', 'conveyance', 'copyrights', 'deed', 'deemed', 'deems', 'delegated', 'designate', 'designating', 'designees', 'desist', 'disallowed', 'disclaim', 'disclaimers', 'disclaims', 'disclose', 'disclosed', 'disclosing', 'disclosure', 'disclosures', 'discretion', 'disqualified', 'disqualify', 'duly', 'effectuate', 'entitle', 'entrant', 'enumerated', 'excludes', 'exercised', 'expressly', 'fide', 'foregoing', 'forfeit', 'forfeited', 'furnish', 'furtherance', 'hereafter', 'hereby', 'herein', 'hereinabove', 'hereinafter', 'hereof', 'hereto', 'heretofore', 'herewith', 'imputed', 'incapacitated', 'incidental', 'indemnify', 'infringe', 'infringing', 'irrevocable', 'irrevocably', 'lawful', 'lawfully', 'legally', 'liable', 'lieu', 'limitation', 'materially', 'nonpublic', 'obligate', 'obligated', 'obligation', 'obligations', 'particulars', 'permissible', 'permitted', 'perpetuity', 'practicable', 'preclude', 'provisionally', 'proviso', 'pursuant', 'recourse', 'relinquished', 'respecting', 'revoke', 'revoking', 'severally', 'shall', 'stipulate', 'stipulated', 'stipulation', 'stipulations', 'subparagraph', 'subsection', 'subsections', 'supercede', 'supersede', 'supersedes', 'tendered', 'terminate', 'terminated', 'termination', 'therefor', 'thereof', 'thereto', 'therewith', 'trademarks', 'transact', 'transacted', 'unauthorised', 'unauthorized', 'unconditionally', 'undersigned', 'undertakings', 'unlawfully', 'vested', 'voided', 'voluntarily', 'waive', 'waiving', 'warranties', 'warrants', 'withheld', 'withhold']

ActingTechSpeak=['accessibility', 'adaptive', 'advanced', 'agile', 'analytics', 'application', 'applications', 'architecture', 'archiving', 'augmented', 'automate', 'automated', 'automates', 'automating', 'automation', 'biometric', 'capabilities', 'capability', 'centralized', 'centric', 'client', 'cloud', 'coding', 'communication', 'compatibility', 'compliant', 'computing', 'connectivity', 'converged', 'customization', 'customized', 'databases', 'datacenter', 'decentralized', 'deploy', 'deploying', 'deployment', 'deployments', 'designing', 'desktops', 'developer', 'developers', 'dynamic', 'ecommerce', 'embedded', 'enable', 'enabled', 'enabler', 'enables', 'enabling', 'enhancements', 'enterprise', 'environments', 'extranet', 'federated', 'framework', 'frameworks', 'functionalities', 'functionality', 'implement', 'implementation', 'implementations', 'implemented', 'infrastructure', 'infrastructures', 'instrumentation', 'integrate', 'integrated', 'integrates', 'integrating', 'integration', 'integrator', 'interactive', 'interoperability', 'intranet', 'intranets', 'intuitive', 'leverage', 'leverages', 'leveraging', 'libraries', 'lifecycle', 'mainframe', 'middleware', 'monetization', 'multimedia', 'networked', 'networking', 'offsite', 'optimisation', 'optimization', 'optimize', 'optimized', 'optimizing', 'paperless', 'personalization', 'platform', 'platforms', 'processes', 'programming', 'proprietary', 'protocols', 'provisioning', 'realtime', 'redundancy', 'roadmap', 'robust', 'rollout', 'scalability', 'scalable', 'seamless', 'seamlessly', 'servers', 'simplified', 'simplifies', 'simplify', 'software', 'solution', 'solutions', 'spreadsheets', 'standalone', 'standardization', 'streamline', 'streamlined', 'structured', 'systems', 'technologies', 'technology', 'throughput', 'tool', 'toolkit', 'tools', 'tracking', 'transactional', 'troubleshooting', 'turnkey', 'unified', 'unstructured', 'usability', 'usage', 'utilising', 'utilize', 'utilizing', 'verticals', 'virtual', 'visualization', 'workflow', 'workstations', 'accelerator', 'activated', 'adapter', 'adapters', 'allows', 'analyzer', 'apparatus', 'array', 'arrays', 'automatic', 'automatically', 'auxiliary', 'capable', 'charging', 'chip', 'circuit', 'circuits', 'cluster', 'coded', 'compatible', 'component', 'components', 'computer', 'computers', 'configuration', 'configurations', 'configured', 'connect', 'connected', 'connection', 'connections', 'connector', 'console', 'control', 'controllable', 'controlled', 'controller', 'controllers', 'controls', 'conversion', 'converters', 'converting', 'converts', 'detect', 'detection', 'detects', 'device', 'devices', 'diagram', 'disk', 'disks', 'driver', 'drives', 'dual', 'electronic', 'external', 'function', 'functions', 'generating', 'hardware', 'input', 'inputs', 'interface', 'interfaced', 'interfaces', 'interfacing', 'internal', 'keypad', 'keys', 'latency', 'load', 'loading', 'machine', 'machines', 'manual', 'mechanism', 'memory', 'metering', 'method', 'microprocessor', 'mode', 'modes', 'module', 'modules', 'monitor', 'monitors', 'multiple', 'operable', 'operate', 'operating', 'operation', 'operator', 'optional', 'output', 'outputs', 'passive', 'peripherals', 'playback', 'port', 'portable', 'ports', 'processing', 'processor', 'processors', 'programmable', 'programmed', 'redundant', 'registers', 'remote', 'remotely', 'scan', 'scanner', 'scanners', 'scanning', 'schematic', 'selector', 'sensing', 'sensor', 'sensors', 'serial', 'socket', 'sockets', 'specification', 'storage', 'storing', 'subsystem', 'subsystems', 'switch', 'switchable', 'switches', 'switching', 'synchronization', 'synchronized', 'synchronizing', 'synchronous', 'system', 'terminal', 'terminals', 'tester', 'timer', 'timers', 'transfer', 'unit', 'units', 'workstation']

ActingProjectManagement=['above', 'according', 'action', 'active', 'added', 'adding', 'addition', 'additional', 'advance', 'after', 'another', 'anticipated', 'appearance', 'appears', 'as', 'at', 'available', 'based', 'before', 'beginning', 'below', 'between', 'board', 'branch', 'calendar', 'case', 'charge', 'closed', 'closing', 'combined', 'complete', 'completed', 'completing', 'conjunction', 'count', 'course', 'cover', 'coverage', 'covered', 'covering', 'created', 'current', 'currently', 'daily', 'date', 'dates', 'day', 'days', 'direct', 'during', 'duty', 'each', 'end', 'ending', 'entering', 'entire', 'entry', 'every', 'example', 'expanded', 'extended', 'extension', 'field', 'finished', 'first', 'following', 'follows', 'force', 'form', 'found', 'from', 'full', 'further', 'given', 'group', 'having', 'home', 'hour', 'hours', 'include', 'included', 'includes', 'including', 'initial', 'issue', 'key', 'last', 'latest', 'launch', 'lead', 'least', 'lifetime', 'made', 'main', 'major', 'mark', 'marked', 'marking', 'marks', 'month', 'months', 'name', 'new', 'next', 'number', 'offered', 'only', 'opening', 'original', 'part', 'past', 'period', 'permanent', 'plan', 'planned', 'plus', 'position', 'preparing', 'present', 'press', 'previous', 'primary', 'prior', 'process', 'program', 'provided', 'received', 'recent', 'recommended', 'record', 'regular', 'release', 'released', 'releases', 'reserve', 'return', 's', 'same', 'schedule', 'scheduled', 'second', 'selected', 'separate', 'set', 'setting', 'short', 'showing', 'shown', 'shows', 'sign', 'signs', 'single', 'special', 'stage', 'stages', 'starting', 'state', 'taken', 'target', 'than', 'third', 'this', 'through', 'throughout', 'time', 'times', 'today', 'track', 'twice', 'under', 'until', 'used', 'view', 'visits', 'week', 'weeks', 'where', 'which', 'while', 'within', 'working', 'works', 'year', 'accesses', 'accessing', 'alias', 'aliases', 'annotations', 'assign', 'assigning', 'attribute', 'attributes', 'authenticated', 'authentication', 'breakpoint', 'buffer', 'byte', 'bytes', 'cache', 'caching', 'command', 'commands', 'constructors', 'dataset', 'datastore', 'decode', 'defaults', 'deletion', 'descriptors', 'disallow', 'duplicate', 'duplicates', 'dynamically', 'encoding', 'encrypted', 'encryption', 'execute', 'executing', 'failover', 'fallback', 'fetch', 'fetching', 'filtering', 'formatted', 'granularity', 'grouping', 'handler', 'hash', 'header', 'headers', 'hierarchical', 'hypertext', 'identifier', 'ids', 'indexing', 'initialize', 'inputted', 'inputting', 'invalid', 'invoke', 'invoked', 'locale', 'logging', 'logins', 'lookup', 'mapped', 'mapper', 'mappings', 'mirroring', 'multicast', 'multipart', 'namespace', 'node', 'nodes', 'object', 'objects', 'octet', 'offload', 'overflow', 'override', 'overrides', 'packet', 'packets', 'paged', 'paging', 'parameter', 'parsing', 'passwords', 'plaintext', 'pointer', 'pointers', 'populate', 'prefix', 'protocol', 'proxies', 'proxy', 'pseudo', 'queried', 'queries', 'query', 'querying', 'queue', 'queued', 'reachable', 'redirect', 'redirection', 'redirects', 'remapping', 'reorder', 'retransmission', 'retransmit', 'retransmitted', 'retrieval', 'retrieve', 'retrieving', 'routing', 'scheduler', 'schema', 'schemas', 'scoped', 'sender', 'servlet', 'servlets', 'singleton', 'sorting', 'specified', 'specifies', 'specify', 'specifying', 'stateful', 'static', 'string', 'subclass', 'syntax', 'tempdb', 'terabytes', 'timeout', 'timestamp', 'token', 'tokens', 'unreachable', 'validations', 'verifier', 'windowing']



# boost bags with cosine distance from full glove data set
from tqdm import tqdm
import string
embeddings_index = {}
f = open(GLOVE_DATASET_PATH)
#print(GLOVE_DATASET_PATH)
word_counter = 0
for line in tqdm(f):
  values = line.split()
  word = values[0]
  # difference here as we don't intersect words, we take most of them
  if (word.islower() and word.isalpha() and '@' not in word):
       # work with smaller list of vectors
    #print(word)
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
  word_counter += 1
f.close()

print('Found %s word vectors matching enron data set.' % len(embeddings_index))
print('Total words in GloVe data set: %s' % word_counter)

# create a dataframe using the embedded vectors and attach the key word as row header
glove_dataframe = pd.DataFrame(embeddings_index)
glove_dataframe = pd.DataFrame.transpose(glove_dataframe)

#departments = [LEGAL, COMMUICATIONS, SECURITY_SPAM_ALERTS, SUPPORT, ENERGY_DESK, SALES_DEPARTMENT]

#departments = [Networking, Restraining, Facilitating, Motivating, PhysicalAction, Praise, Criticism]

categories = [
LeadingNetworking,
LeadingRestraining,
LeadingFacilitating,
LeadingMotivating,
LeadingPhysicalAction,
LeadingPraise,
LeadingCriticism,
ThinkingLearningOrCreative,
ThinkingSpiritual,
ThinkingNobility,
ThinkingAmbiguity,
ThinkingCurrentAffairs,
ThinkingAnalytical,
SpeakingFormality,
SpeakingPop,
SpeakingGeek,
SpeakingCasualAndFamily,
SpeakingMachismo,
SpeakingHumanity,
SpeakingDramatic,
SpeakingBanter,
ActingUrgency,
ActingIndustryJargon,
ActingOfficialeseAndLegalese,
ActingTechSpeak,
ActingProjectManagement,
SpeakingProfanity
]

temp_matrix = pd.DataFrame.as_matrix(glove_dataframe)
import scipy
import scipy.spatial

vocab_boost_count = 5
for group_id in range(len(categories)):
  print('Working bag number:', str(group_id))
  glove_dataframe_temp = glove_dataframe.copy()
  vocab = []
  for word in categories[group_id]:
    print(word)
    vocab.append(word)
    cos_dist_rez = scipy.spatial.distance.cdist(temp_matrix, np.array(glove_dataframe.loc[word])[np.newaxis,:], metric='cosine')
    # find closest words to help
    glove_dataframe_temp['cdist'] = cos_dist_rez
    glove_dataframe_temp = glove_dataframe_temp.sort_values(['cdist'], ascending=[1])
    vocab = vocab + list(glove_dataframe_temp.head(vocab_boost_count).index)
  # replace boosted set to old department group and remove duplicates
  categories[group_id] = list(set(vocab))

# save final objects to disk
import cPickle as pickle
with open('full_bags.pk', 'wb') as handle:
  pickle.dump(categories, handle)




#####################################################################
# Create features of word counts for each department in each email
#####################################################################
print('Opening categories from disk')

import cPickle as pickle
from tqdm import tqdm

with open('full_bags.pk', 'rb') as handle:
    categories = pickle.load(handle)


# Remove high frequency words that don't give strong category signal

stopwords = ['a', 'about', 'above', 'across', 'after', 'afterwards']
stopwords += ['again', 'against', 'all', 'almost', 'alone', 'along']
stopwords += ['already', 'also', 'although', 'always', 'am', 'among']
stopwords += ['amongst', 'amoungst', 'amount', 'an', 'and', 'another']
stopwords += ['any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere']
stopwords += ['are', 'around', 'as', 'at', 'back', 'be', 'became']
stopwords += ['because', 'become', 'becomes', 'becoming', 'been']
stopwords += ['before', 'beforehand', 'behind', 'being', 'below']
stopwords += ['beside', 'besides', 'between', 'beyond', 'bill', 'both']
stopwords += ['bottom', 'but', 'by', 'call', 'can', 'cannot', 'cant']
stopwords += ['co', 'computer', 'con', 'could', 'couldnt', 'cry', 'de']
stopwords += ['describe', 'detail', 'did', 'do', 'done', 'down', 'due']
stopwords += ['during', 'each', 'eg', 'eight', 'either', 'eleven', 'else']
stopwords += ['elsewhere', 'empty', 'enough', 'etc', 'even', 'ever']
stopwords += ['every', 'everyone', 'everything', 'everywhere', 'except']
stopwords += ['few', 'fifteen', 'fifty', 'fill', 'find', 'fire', 'first']
stopwords += ['five', 'for', 'former', 'formerly', 'forty', 'found']
stopwords += ['four', 'from', 'front', 'full', 'further', 'get', 'give']
stopwords += ['go', 'had', 'has', 'hasnt', 'have', 'he', 'hence', 'her']
stopwords += ['here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers']
stopwords += ['herself', 'him', 'himself', 'his', 'how', 'however']
stopwords += ['hundred', 'i', 'ie', 'if', 'in', 'inc', 'indeed']
stopwords += ['interest', 'into', 'is', 'it', 'its', 'itself', 'keep']
stopwords += ['last', 'latter', 'latterly', 'least', 'less', 'ltd', 'made']
stopwords += ['many', 'may', 'me', 'meanwhile', 'might', 'mill', 'mine']
stopwords += ['more', 'moreover', 'most', 'mostly', 'move', 'much']
stopwords += ['must', 'my', 'myself', 'name', 'namely', 'neither', 'never']
stopwords += ['nevertheless', 'next', 'nine', 'no', 'nobody', 'none']
stopwords += ['noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'of']
stopwords += ['off', 'often', 'on','once', 'one', 'only', 'onto', 'or']
stopwords += ['other', 'others', 'otherwise', 'our', 'ours', 'ourselves']
stopwords += ['out', 'over', 'own', 'part', 'per', 'perhaps', 'please']
stopwords += ['put', 'rather', 're', 's', 'same', 'see', 'seem', 'seemed']
stopwords += ['seeming', 'seems', 'serious', 'several', 'she', 'should']
stopwords += ['show', 'side', 'since', 'sincere', 'six', 'sixty', 'so']
stopwords += ['some', 'somehow', 'someone', 'something', 'sometime']
stopwords += ['sometimes', 'somewhere', 'still', 'such', 'system', 'take']
stopwords += ['ten', 'than', 'that', 'the', 'their', 'them', 'themselves']
stopwords += ['then', 'thence', 'there', 'thereafter', 'thereby']
stopwords += ['therefore', 'therein', 'thereupon', 'these', 'they']
stopwords += ['thick', 'thin', 'third', 'this', 'those', 'though', 'three']
stopwords += ['three', 'through', 'throughout', 'thru', 'thus', 'to']
stopwords += ['together', 'too', 'top', 'toward', 'towards', 'twelve']
stopwords += ['twenty', 'two', 'un', 'under', 'until', 'up', 'upon']
stopwords += ['us', 'very', 'via', 'was', 'we', 'well', 'were', 'what']
stopwords += ['whatever', 'when', 'whence', 'whenever', 'where']
stopwords += ['whereafter', 'whereas', 'whereby', 'wherein', 'whereupon']
stopwords += ['wherever', 'whether', 'which', 'while', 'whither', 'who']
stopwords += ['whoever', 'whole', 'whom', 'whose', 'why', 'will', 'with']
stopwords += ['within', 'without', 'would', 'yet', 'you', 'your']
stopwords += ['yours', 'yourself', 'yourselves']
#Add stopwords determined by in house analysis
stopwords += ['the', 'to', 'and', 'of', 'a', 'enron', 'in', 'for', 'on', 'i', 'is', 'com', 'you', 'ect', 'td', 'that', 'this', 's', 'be', 'at', 'we', 'from', 'will', 'it', 'have', 'are', 'by', 'with', 'font', 'or', 'as', 'http', 'if', 'your', 'hou', 'pm', 'not', 'am', 'subject', 'please', 'b', 'e', 'tr', 're', 'power', 'an', 'me', 'cc', 'all', 'any', 'width', 'our', 'was', 't', 'can', 'align', 'would', 'table', 'has', 'image', 'www', 'corp', 'they', 'right', 'no', 'data', 'time', 'he', 'may', 'up', 'class', 'size', 'know', 'thanks', 'out', 'message', 'm', 'do', 'there', 'new', 'but', 'sent', 'br', 'go', 'about', 'get', 'so', 'more', 'href', 'energy', 'border', 'email', 'said', 'one', 'gas', 'what', 'day', 'which', 'call', 'my', 'should', 'let', 'dbcaps', 'information', 'original', 'forwarded', 'face', 'been', 'also', 'need', 'arial', 'operation', 'helvetica', 'when', 'other', 'contact', 'p', 'california', 'could', 'mail', 'cellpadding', 'us', 'their', 'some', 'houston', 'see', 'just', 'center', '.', 'company', 'these', 'only', 'alias', 'development', 'state', 'its', 'cellspacing', 'first', 'f', 'd', 'fax', 'provantage', 'over', 'mark', 'market', 'like', 'color', 'year', 'group', 'date', 'business', 'memory', 'were', 'meeting', 'attached', 'n', 'insufficient', 'jeff', 'here', 'how', 'today', 'c', 'week', 'price', 'london', 'l', 'x', 'his', 'free', 'deal', 'who', 'take', 'now', 'nl', 'had', 'trading', 'news', 'na', 'next', 'friday', 'them', 'wednesday', 'monday', 'contract', 'questions', 'office', 'work', 'team', 'sat', 'option', 'through', 'sportsline', 'j', 'into', 'bgcolor', 'back', 'two', 'ecom', 'want', 'scheduled', 'review', 'last', 'thru', 'make', 'agreement', 'than', 'th', 'r', 'help', 'john', 'credit', 'w', 'click', 'after', 'she', 'emsn', 'src', 'electricity', 'service', 'request', 'process', 'img', 'database', 'send', 'random', 'htm', 'forward', 'use', 'report', 'don', 'project', 'order', 'going', 'following', 'nbsp', 'name', 'june', 'think', 'system', 'people', 'issue', 'bna', 'start', 'risk', 'best', 'well', 'lon', 'her', 'football', 'vince', 'thursday', 'number', 'legal', 'kay', 'jones', 'gif', 'did', 'll', 'key', 'ees', 'verdana', 'value', 'ena', 'change', 'below', 'under', 'thank', 'height', 'companies', 'tuesday', 'pt', 'o', 'north', 'net', 'fantasy', 'because', 'list', 'where', 'support', 'since', 'plan', 'owner', 'end', 'before', 'access', 'then', 'set', 'script', 'provide', 'mp', 'league', 'during', 'closed', 'approval', 'utilities', 'outages', 'off', 'good', 'give', 'dpc', 'between', 'place', 'mike', 'issues', 'et', 'chris', 'cannot', 'inc', 'file', 'david', 'per', 'page', 'id', 'days', 'comments', 'bige', 'until', 'services', 'most', 'him', 'fw', 'fri', 'error', 'discuss', 'smith', 'received', 'october', 'middle', 'k', 'easp', 'ct', 'being', 'management', 'generation', 'elink', 'available', 'america', 'address', 'top', 'sure', 'made', 'left', 'april', 'site', 'michael', 'look', 'doc', 'co', 'bill', 'very', 've', 'regarding', 'purchase', 'internet', 'home', 'financial', 'each', 'buy', 'v', 'those', 'scripts', 'scott', 'point', 'letter', 'dll', 'copy', 'conference', 'both', 'based', 'while', 'resources', 'markets', 'images', 'valign', 'form', 'daily', 'web', 'tomorrow', 'texas', 'plant', 'operations', 'much', 'intended', 'include', 'got', 'fp', 'whether', 'way', 'still', 'small', 'richard', 'pipeline', 'phone', 'paul', 'mary', 'hope', 'does', 'demand', 'corporate', 'changes', 'board', 'against', 'sunday', 'mseb', 'million', 'lenders', 'however', 'government', 'draft', 'current', 'west', 'updated', 'skimoguls', 'receive', 'players', 'meet', 'fuel', 'federal', 'contracts', 'same', 'pay', 'options', 'find', 'ews', 'communications', 'capacity', 'within', 'steve', 'server', 'performance', 'perform', 'january', 'h', 'due', 'distribution', 'customers', 'city', 'above', 'vacation', 'u', 'supply', 'summer', 'purpose', 'long']



def removeStopwords(wordlist, stopwordsList):
    return [w for w in wordlist if w not in stopwordsList]

print('Removing stop words from categories')

for group_id in range(len(categories)):
  print('Working bag number:', str(group_id))
  print(categories[group_id])
  categories[group_id] = removeStopwords(categories[group_id], stopwords)
#  input("Press Enter to see reduced category")
  print(categories[group_id])


# loop through all emails and count group words in each raw text
words_groups = []
for group_id in range(len(categories)):
  work_group = []
  print('Working bag number:', str(group_id))
  top_words = categories[group_id]
  for index, row in tqdm(emails_sample_df.iterrows()):
    text = (row["Subject"] + " " + row["Content"])
    work_group.append(len(set(top_words) & set(text.split())))
    #work_group.append(len([w for w in text.split() if w in set(top_words)]))

  words_groups.append(work_group)

# loop through all emails and count group words in each raw text
full_words_groups = []
for group_id in range(len(categories)):
  work_group = []
  print('Working bag number:', str(group_id))
  top_words = categories[group_id]
  for index, row in tqdm(emails_sample_df.iterrows()):
    text = (row["Subject"] + " " + row["Content"])
    work_group.append( ' '.join(set(top_words) & set(text.split())) )
    #work_group.append(len([w for w in text.split() if w in set(top_words)]))

  full_words_groups.append(work_group)

"""
# loop through all emails and count group words in each raw text
full_words_groups = []
for group_id in range(len(categories)):
  work_group_2 = []
  print('Working bag number:', str(group_id))
  top_words_2 = categories[group_id]
  for index, row in tqdm(emails_sample_df.iterrows()):
    text = (row["Subject"] + " " + index["Content"])
    work_group_2.append( str(set(top_words_2) & set(text.split())) )
    #work_group.append(len([w for w in text.split() if w in set(top_words)]))

  full_words_groups.append(work_group_2)
"""

print(full_words_groups)



# count emails per category group and feature engineering



raw_text = []
subject = []
subject_length = []

subject_word_count = []
content_length = []
content_word_count = []
is_am_list = []
is_weekday_list = []
date_list = []
group_LeadingNetworking = []
group_LeadingRestraining = []
group_LeadingFacilitating = []
group_LeadingMotivating = []
group_LeadingPhysicalAction = []
group_LeadingPraise = []
group_LeadingCriticism = []
group_ThinkingLearningOrCreative = []
group_ThinkingSpiritual = []
group_ThinkingNobility = []
group_ThinkingAmbiguity = []
group_ThinkingCurrentAffairs = []
group_ThinkingAnalytical = []
group_SpeakingFormality = []
group_SpeakingPop = []
group_SpeakingGeek = []
group_SpeakingCasualAndFamily = []
group_SpeakingMachismo = []
group_SpeakingHumanity = []
group_SpeakingDramatic = []
group_SpeakingBanter = []
group_ActingUrgency = []
group_ActingIndustryJargon = []
group_ActingOfficialeseAndLegalese = []
group_ActingTechSpeak = []
group_ActingProjectManagement = []
group_SpeakingProfanity = []

group_LeadingNetworking_Words = []
group_LeadingRestraining_Words = []
group_LeadingFacilitating_Words = []
group_LeadingMotivating_Words = []
group_LeadingPhysicalAction_Words = []
group_LeadingPraise_Words = []
group_LeadingCriticism_Words = []
group_ThinkingLearningOrCreative_Words = []
group_ThinkingSpiritual_Words = []
group_ThinkingNobility_Words = []
group_ThinkingAmbiguity_Words = []
group_ThinkingCurrentAffairs_Words = []
group_ThinkingAnalytical_Words = []
group_SpeakingFormality_Words = []
group_SpeakingPop_Words = []
group_SpeakingGeek_Words = []
group_SpeakingCasualAndFamily_Words = []
group_SpeakingMachismo_Words = []
group_SpeakingHumanity_Words = []
group_SpeakingDramatic_Words = []
group_SpeakingBanter_Words = []
group_ActingUrgency_Words = []
group_ActingIndustryJargon_Words = []
group_ActingOfficialeseAndLegalese_Words = []
group_ActingTechSpeak_Words = []
group_ActingProjectManagement_Words = []
group_SpeakingProfanity = []

final_outcome = []


emails_sample_df['Subject'].fillna('', inplace=True)
emails_sample_df['Date'] = pd.to_datetime(emails_sample_df['Date'], infer_datetime_format=True)



counter = 0
for index, row in tqdm(emails_sample_df.iterrows()):
  raw_text.append([row["Subject"] + " " + row["Content"]])
  subject.append([row["Subject"]])
  group_LeadingNetworking.append(words_groups[0][counter])
  group_LeadingRestraining.append(words_groups[1][counter])
  group_LeadingFacilitating.append(words_groups[2][counter])
  group_LeadingMotivating.append(words_groups[3][counter])
  group_LeadingPhysicalAction.append(words_groups[4][counter])
  group_LeadingPraise.append(words_groups[5][counter])
  group_LeadingCriticism.append(words_groups[6][counter])
  group_ThinkingLearningOrCreative.append(words_groups[7][counter])
  group_ThinkingSpiritual.append(words_groups[8][counter])
  group_ThinkingNobility.append(words_groups[9][counter])
  group_ThinkingAmbiguity.append(words_groups[10][counter])
  group_ThinkingCurrentAffairs.append(words_groups[11][counter])
  group_ThinkingAnalytical.append(words_groups[12][counter])
  group_SpeakingFormality.append(words_groups[13][counter])
  group_SpeakingPop.append(words_groups[14][counter])
  group_SpeakingGeek.append(words_groups[15][counter])
  group_SpeakingCasualAndFamily.append(words_groups[16][counter])
  group_SpeakingMachismo.append(words_groups[17][counter])
  group_SpeakingHumanity.append(words_groups[18][counter])
  group_SpeakingDramatic.append(words_groups[19][counter])
  group_SpeakingBanter.append(words_groups[20][counter])
  group_ActingUrgency.append(words_groups[21][counter])
  group_ActingIndustryJargon.append(words_groups[22][counter])
  group_ActingOfficialeseAndLegalese.append(words_groups[23][counter])
  group_ActingTechSpeak.append(words_groups[24][counter])
  group_ActingProjectManagement.append(words_groups[25][counter])
  group_SpeakingProfanity.append(words_groups[26[counter]])
  group_LeadingNetworking_Words.append(full_words_groups[0][counter])
  group_LeadingRestraining_Words.append(full_words_groups[1][counter])
  group_LeadingFacilitating_Words.append(full_words_groups[2][counter])
  group_LeadingMotivating_Words.append(full_words_groups[3][counter])
  group_LeadingPhysicalAction_Words.append(full_words_groups[4][counter])
  group_LeadingPraise_Words.append(full_words_groups[5][counter])
  group_LeadingCriticism_Words.append(full_words_groups[6][counter])
  group_ThinkingLearningOrCreative_Words.append(full_words_groups[7][counter])
  group_ThinkingSpiritual_Words.append(full_words_groups[8][counter])
  group_ThinkingNobility_Words.append(full_words_groups[9][counter])
  group_ThinkingAmbiguity_Words.append(full_words_groups[10][counter])
  group_ThinkingCurrentAffairs_Words.append(full_words_groups[11][counter])
  group_ThinkingAnalytical_Words.append(full_words_groups[12][counter])
  group_SpeakingFormality_Words.append(full_words_groups[13][counter])
  group_SpeakingPop_Words.append(full_words_groups[14][counter])
  group_SpeakingGeek_Words.append(full_words_groups[15][counter])
  group_SpeakingCasualAndFamily_Words.append(full_words_groups[16][counter])
  group_SpeakingMachismo_Words.append(full_words_groups[17][counter])
  group_SpeakingHumanity_Words.append(full_words_groups[18][counter])
  group_SpeakingDramatic_Words.append(full_words_groups[19][counter])
  group_SpeakingBanter_Words.append(full_words_groups[20][counter])
  group_ActingUrgency_Words.append(full_words_groups[21][counter])
  group_ActingIndustryJargon_Words.append(full_words_groups[22][counter])
  group_ActingOfficialeseAndLegalese_Words.append(full_words_groups[23][counter])
  group_ActingTechSpeak_Words.append(full_words_groups[24][counter])
  group_ActingProjectManagement_Words.append(full_words_groups[25][counter])
  group_SpeakingProfanity_Words.append(full_words[26[counter]])
  outcome_tots = [words_groups[0][counter], words_groups[1][counter], words_groups[2][counter],
    words_groups[3][counter], words_groups[4][counter], words_groups[5][counter], words_groups[6][counter],
    words_groups[7][counter], words_groups[8][counter], words_groups[9][counter], words_groups[10][counter],
    words_groups[11][counter], words_groups[12][counter], words_groups[13][counter],words_groups[14][counter],
    words_groups[15][counter], words_groups[16][counter], words_groups[17][counter], words_groups[18][counter],
    words_groups[19][counter], words_groups[20][counter], words_groups[21][counter], words_groups[22][counter],
    words_groups[23][counter], words_groups[24][counter], words_groups[25][counter], words_groups[26][counter]]
  final_outcome.append(outcome_tots.index(max(outcome_tots)))

  #print("Testing FINALOUTCOME")
  #print(final_outcome)


  subject_length.append(len(row['Subject']))
  subject_word_count.append(len(row['Subject'].split()))
  content_length.append(len(row['Content']))
  content_word_count.append(len(row['Content'].split()))
  dt = row['Date']
  date_list.append(dt)
  is_am = 'no'
  if (dt.time() < datetime.time(12)): is_am = 'yes'
  is_am_list.append(is_am)
  is_weekday = 'no'
  if (dt.weekday() < 6): is_weekday = 'yes'
  is_weekday_list.append(is_weekday)
  counter += 1

# add simple engineered features?
training_set = pd.DataFrame({
              "raw_text":raw_text,
              "group_LeadingNetworking":group_LeadingNetworking,
              "group_LeadingRestraining":group_LeadingRestraining,
              "group_LeadingFacilitating":group_LeadingFacilitating,
              "group_LeadingMotivating":group_LeadingMotivating,
              "group_LeadingPhysicalAction":group_LeadingPhysicalAction,
              "group_LeadingPraise":group_LeadingPraise,
              "group_LeadingCriticism":group_LeadingCriticism,
              "group_ThinkingLearningOrCreative":group_ThinkingLearningOrCreative,
              "group_ThinkingSpiritual":group_ThinkingSpiritual,
              "group_ThinkingNobility":group_ThinkingNobility,
              "group_ThinkingAmbiguity":group_ThinkingAmbiguity,
              "group_ThinkingCurrentAffairs":group_ThinkingCurrentAffairs,
              "group_ThinkingAnalytical":group_ThinkingAnalytical,
              "group_SpeakingFormality":group_SpeakingFormality,
              "group_SpeakingPop":group_SpeakingPop,
              "group_SpeakingGeek":group_SpeakingGeek,
              "group_SpeakingCasualAndFamily":group_SpeakingCasualAndFamily,
              "group_SpeakingMachismo":group_SpeakingMachismo,
              "group_SpeakingHumanity":group_SpeakingHumanity,
              "group_SpeakingDramatic":group_SpeakingDramatic,
              "group_SpeakingBanter":group_SpeakingBanter,
              "group_ActingUrgency":group_ActingUrgency,
              "group_ActingIndustryJargon":group_ActingIndustryJargon,
              "group_ActingOfficialeseAndLegalese":group_ActingOfficialeseAndLegalese,
              "group_ActingTechSpeak":group_ActingTechSpeak,
              "group_ActingProjectManagement":group_ActingProjectManagement,
              "group_SpeakingProfanity":group_SpeakingProfanity,
              "subject_length":subject_length,
              "subject_word_count":subject_word_count,
              "content_length":content_length,
              "content_word_count":content_word_count,
              "is_AM":is_am_list,
              "is_weekday":is_weekday_list,
              "outcome":final_outcome})

analysis_set = pd.DataFrame({
                "raw_text":raw_text,
                "subject":subject,
                "date":date_list,
                "group_LeadingNetworking":group_LeadingNetworking,
                "group_LeadingRestraining":group_LeadingRestraining,
                "group_LeadingFacilitating":group_LeadingFacilitating,
                "group_LeadingMotivating":group_LeadingMotivating,
                "group_LeadingPhysicalAction":group_LeadingPhysicalAction,
                "group_LeadingPraise":group_LeadingPraise,
                "group_LeadingCriticism":group_LeadingCriticism,
                "group_ThinkingLearningOrCreative":group_ThinkingLearningOrCreative,
                "group_ThinkingSpiritual":group_ThinkingSpiritual,
                "group_ThinkingNobility":group_ThinkingNobility,
                "group_ThinkingAmbiguity":group_ThinkingAmbiguity,
                "group_ThinkingCurrentAffairs":group_ThinkingCurrentAffairs,
                "group_ThinkingAnalytical":group_ThinkingAnalytical,
                "group_SpeakingFormality":group_SpeakingFormality,
                "group_SpeakingPop":group_SpeakingPop,
                "group_SpeakingGeek":group_SpeakingGeek,
                "group_SpeakingCasualAndFamily":group_SpeakingCasualAndFamily,
                "group_SpeakingMachismo":group_SpeakingMachismo,
                "group_SpeakingHumanity":group_SpeakingHumanity,
                "group_SpeakingDramatic":group_SpeakingDramatic,
                "group_SpeakingBanter":group_SpeakingBanter,
                "group_ActingUrgency":group_ActingUrgency,
                "group_ActingIndustryJargon":group_ActingIndustryJargon,
                "group_ActingOfficialeseAndLegalese":group_ActingOfficialeseAndLegalese,
                "group_ActingTechSpeak":group_ActingTechSpeak,
                "group_SpeakingProfanity":group_SpeakingProfanity,
                "group_ActingProjectManagement":group_ActingProjectManagement,
                "group_LeadingNetworking_Words":group_LeadingNetworking_Words,
                "group_LeadingRestraining_Words":group_LeadingRestraining_Words,
                "group_LeadingFacilitating_Words":group_LeadingFacilitating_Words,
                "group_LeadingMotivating_Words":group_LeadingMotivating_Words,
                "group_LeadingPhysicalAction_Words":group_LeadingPhysicalAction_Words,
                "group_LeadingPraise_Words":group_LeadingPraise_Words,
                "group_LeadingCriticism_Words":group_LeadingCriticism_Words,
                "group_ThinkingLearningOrCreative_Words":group_ThinkingLearningOrCreative_Words,
                "group_ThinkingSpiritual_Words":group_ThinkingSpiritual_Words,
                "group_ThinkingNobility_Words":group_ThinkingNobility_Words,
                "group_ThinkingAmbiguity_Words":group_ThinkingAmbiguity_Words,
                "group_ThinkingCurrentAffairs_Words":group_ThinkingCurrentAffairs_Words,
                "group_ThinkingAnalytical_Words":group_ThinkingAnalytical_Words,
                "group_SpeakingFormality_Words":group_SpeakingFormality_Words,
                "group_SpeakingPop_Words":group_SpeakingPop_Words,
                "group_SpeakingGeek_Words":group_SpeakingGeek_Words,
                "group_SpeakingCasualAndFamily_Words":group_SpeakingCasualAndFamily_Words,
                "group_SpeakingMachismo_Words":group_SpeakingMachismo_Words,
                "group_SpeakingHumanity_Words":group_SpeakingHumanity_Words,
                "group_SpeakingDramatic_Words":group_SpeakingDramatic_Words,
                "group_SpeakingBanter_Words":group_SpeakingBanter_Words,
                "group_ActingUrgency_Words":group_ActingUrgency_Words,
                "group_ActingIndustryJargon_Words":group_ActingIndustryJargon_Words,
                "group_ActingOfficialeseAndLegalese_Words":group_ActingOfficialeseAndLegalese_Words,
                "group_ActingTechSpeak_Words":group_ActingTechSpeak_Words,
                "group_ActingProjectManagement_Words":group_ActingProjectManagement_Words,
                "group_SpeakingProfanity_Words":group_SpeakingProfanity_Words,
                "subject_length":subject_length,
                "subject_word_count":subject_word_count,
                "content_length":content_length,
                "content_word_count":content_word_count,
                "is_AM":is_am_list,
                "is_weekday":is_weekday_list,
                "outcome":final_outcome})

print(analysis_set)
analysis_set.to_csv('PaddleEmailAnalysis.csv', sep='\t', encoding='utf-8')

# remove all emails that have all zeros (i.e. not from any of required categories)
training_set = training_set[(training_set.group_LeadingNetworking > 0) |
              (training_set.group_LeadingRestraining > 0) |
              (training_set.group_LeadingFacilitating > 0) |
              (training_set.group_LeadingMotivating > 0) |
              (training_set.group_LeadingPhysicalAction > 0) |
              (training_set.group_LeadingPraise > 0) |
              (training_set.group_LeadingCriticism > 0) |
              (training_set.group_ThinkingLearningOrCreative > 0) |
              (training_set.group_ThinkingSpiritual > 0) |
              (training_set.group_ThinkingNobility > 0) |
              (training_set.group_ThinkingAmbiguity > 0) |
              (training_set.group_ThinkingCurrentAffairs > 0) |
              (training_set.group_ThinkingAnalytical > 0) |
              (training_set.group_SpeakingFormality > 0) |
              (training_set.group_SpeakingPop > 0) |
              (training_set.group_SpeakingGeek > 0) |
              (training_set.group_SpeakingCasualAndFamily > 0) |
              (training_set.group_SpeakingMachismo > 0) |
              (training_set.group_SpeakingHumanity > 0) |
              (training_set.group_SpeakingDramatic > 0) |
              (training_set.group_SpeakingBanter > 0) |
              (training_set.group_ActingUrgency > 0) |
              (training_set.group_ActingIndustryJargon > 0) |
              (training_set.group_ActingOfficialeseAndLegalese > 0) |
              (training_set.group_ActingTechSpeak > 0) |
              (training_set.group_ActingProjectManagement > 0) |
              (training_set.group_SpeakingProfanity > 0)
              ]
print(len(training_set))

# save extractions to file
training_set.to_csv('enron_classification_df.csv', index=False, header=True)


####################################################
# TensorFlow Deep Classifier
####################################################
# https://www.tensorflow.org/api_docs/python/tf/contrib/learn/DNNClassifier
# create a wide and deep model and also predict a few entries

model_ready_data = pd.read_csv('enron_classification_df.csv')

# https://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train-validation-and-test
# (60% - train set, 20% - validation set, 20% - test set)
df_train, df_test, df_val = np.split(model_ready_data.sample(frac=1), [int(.6*len(model_ready_data)), int(.8*len(model_ready_data))])

group_LeadingNetworking = []
group_LeadingRestraining = []
group_LeadingFacilitating = []
group_LeadingMotivating = []
group_LeadingPhysicalAction = []
group_LeadingPraise = []
group_LeadingCriticism = []
group_ThinkingLearningOrCreative = []
group_ThinkingSpiritual = []
group_ThinkingNobility = []
group_ThinkingAmbiguity = []
group_ThinkingCurrentAffairs = []
group_ThinkingAnalytical = []
group_SpeakingFormality = []
group_SpeakingPop = []
group_SpeakingGeek = []
group_SpeakingCasualAndFamily = []
group_SpeakingMachismo = []
group_SpeakingHumanity = []
group_SpeakingDramatic = []
group_SpeakingBanter = []
group_ActingUrgency = []
group_ActingIndustryJargon = []
group_ActingOfficialeseAndLegalese = []
group_ActingTechSpeak = []
group_ActingProjectManagement = []
group_SpeakingProfanity = []


# Continuous base columns
content_length = tf.contrib.layers.real_valued_column("content_length")
content_word_count = tf.contrib.layers.real_valued_column("content_word_count")
subject_length = tf.contrib.layers.real_valued_column("subject_length")
subject_word_count = tf.contrib.layers.real_valued_column("subject_word_count")
group_LeadingNetworking = tf.contrib.layers.real_valued_column("group_LeadingNetworking")
group_LeadingRestraining = tf.contrib.layers.real_valued_column("group_LeadingRestraining")
group_LeadingFacilitating = tf.contrib.layers.real_valued_column("group_LeadingFacilitating")
group_LeadingMotivating = tf.contrib.layers.real_valued_column("group_LeadingMotivating")
group_LeadingPhysicalAction = tf.contrib.layers.real_valued_column("group_LeadingPhysicalAction")
group_LeadingPraise = tf.contrib.layers.real_valued_column("group_LeadingPraise")
group_LeadingCriticism = tf.contrib.layers.real_valued_column("group_LeadingCriticism")
group_ThinkingLearningOrCreative = tf.contrib.layers.real_valued_column("group_ThinkingLearningOrCreative")
group_ThinkingSpiritual = tf.contrib.layers.real_valued_column("group_ThinkingSpiritual")
group_ThinkingNobility = tf.contrib.layers.real_valued_column("group_ThinkingNobility")
group_ThinkingAmbiguity = tf.contrib.layers.real_valued_column("group_ThinkingAmbiguity")
group_ThinkingCurrentAffairs = tf.contrib.layers.real_valued_column("group_ThinkingCurrentAffairs")
group_ThinkingAnalytical = tf.contrib.layers.real_valued_column("group_ThinkingAnalytical")
group_SpeakingFormality = tf.contrib.layers.real_valued_column("group_SpeakingFormality")
group_SpeakingPop = tf.contrib.layers.real_valued_column("group_SpeakingPop")
group_SpeakingGeek = tf.contrib.layers.real_valued_column("group_SpeakingGeek")
group_SpeakingCasualAndFamily = tf.contrib.layers.real_valued_column("group_SpeakingCasualAndFamily")
group_SpeakingMachismo = tf.contrib.layers.real_valued_column("group_SpeakingMachismo")
group_SpeakingHumanity = tf.contrib.layers.real_valued_column("group_SpeakingHumanity")
group_SpeakingDramatic = tf.contrib.layers.real_valued_column("group_SpeakingDramatic")
group_SpeakingBanter = tf.contrib.layers.real_valued_column("group_SpeakingBanter")
group_ActingUrgency = tf.contrib.layers.real_valued_column("group_ActingUrgency")
group_ActingIndustryJargon = tf.contrib.layers.real_valued_column("group_ActingIndustryJargon")
group_ActingOfficialeseAndLegalese = tf.contrib.layers.real_valued_column("group_ActingOfficialeseAndLegalese")
group_ActingTechSpeak = tf.contrib.layers.real_valued_column("group_ActingTechSpeak")
group_ActingProjectManagement = tf.contrib.layers.real_valued_column("group_ActingProjectManagement")
group_SpeakingProfanity = tf.contrib.layers.real_valued_column("group_SpeakingProfanity")
content_length_bucket = tf.contrib.layers.bucketized_column(content_length, boundaries=[100, 200, 300, 400])
subject_length_bucket = tf.contrib.layers.bucketized_column(subject_length, boundaries=[10,15, 20, 25, 30])

# Categorical base columns
is_AM_sparse_column = tf.contrib.layers.sparse_column_with_keys(column_name="is_AM", keys=["yes", "no"])
# is_AM = tf.contrib.layers.one_hot_column(is_AM_sparse_column)\
is_weekday_sparse_column = tf.contrib.layers.sparse_column_with_keys(column_name="is_weekday", keys=["yes", "no"])
# is_weekday = tf.contrib.layers.one_hot_column(is_weekday_sparse_column)

categorical_columns = [is_AM_sparse_column, is_weekday_sparse_column, content_length_bucket, subject_length_bucket]



deep_columns = [content_length, content_word_count, subject_length, subject_word_count,
               group_LeadingNetworking,
               group_LeadingRestraining,
               group_LeadingFacilitating,
               group_LeadingMotivating,
               group_LeadingPhysicalAction,
               group_LeadingPraise,
               group_LeadingCriticism,
               group_ThinkingLearningOrCreative,
               group_ThinkingSpiritual,
               group_ThinkingNobility,
               group_ThinkingAmbiguity,
               group_ThinkingCurrentAffairs,
               group_ThinkingAnalytical,
               group_SpeakingFormality,
               group_SpeakingPop,
               group_SpeakingGeek,
               group_SpeakingCasualAndFamily,
               group_SpeakingMachismo,
               group_SpeakingHumanity,
               group_SpeakingDramatic,
               group_SpeakingBanter,
               group_ActingUrgency,
               group_ActingIndustryJargon,
               group_ActingOfficialeseAndLegalese,
               group_ActingTechSpeak,
               group_ActingProjectManagement,
               group_SpeakingProfanity]

simple_columns = [group_LeadingNetworking,
               group_LeadingRestraining,
               group_LeadingFacilitating,
               group_LeadingMotivating,
               group_LeadingPhysicalAction,
               group_LeadingPraise,
               group_LeadingCriticism,
               group_ThinkingLearningOrCreative,
               group_ThinkingSpiritual,
               group_ThinkingNobility,
               group_ThinkingAmbiguity,
               group_ThinkingCurrentAffairs,
               group_ThinkingAnalytical,
               group_SpeakingFormality,
               group_SpeakingPop,
               group_SpeakingGeek,
               group_SpeakingCasualAndFamily,
               group_SpeakingMachismo,
               group_SpeakingHumanity,
               group_SpeakingDramatic,
               group_SpeakingBanter,
               group_ActingUrgency,
               group_ActingIndustryJargon,
               group_ActingOfficialeseAndLegalese,
               group_ActingTechSpeak,
               group_ActingProjectManagement,
               group_SpeakingProfanity]

import tempfile
model_dir = tempfile.mkdtemp()
classifier = tf.contrib.learn.DNNClassifier(feature_columns=simple_columns,
                                hidden_units=[20, 20],
                                n_classes=27,
                                model_dir=model_dir,)



# Define the column names for the data sets.
COLUMNS = ['content_length',
 'content_word_count',
 'group_LeadingNetworking',
 'group_LeadingRestraining',
 'group_LeadingFacilitating',
 'group_LeadingMotivating',
 'group_LeadingPhysicalAction',
 'group_LeadingPraise',
 'group_LeadingCriticism',
 'group_ThinkingLearningOrCreative',
 'group_ThinkingSpiritual',
 'group_ThinkingNobility',
 'group_ThinkingAmbiguity',
 'group_ThinkingCurrentAffairs',
 'group_ThinkingAnalytical',
 'group_SpeakingFormality',
 'group_SpeakingPop',
 'group_SpeakingGeek',
 'group_SpeakingCasualAndFamily',
 'group_SpeakingMachismo',
 'group_SpeakingHumanity',
 'group_SpeakingDramatic',
 'group_SpeakingBanter',
 'group_ActingUrgency',
 'group_ActingIndustryJargon',
 'group_ActingOfficialeseAndLegalese',
 'group_ActingTechSpeak',
 'group_ActingProjectManagement',
 'group_SpeakingProfanity',
 'is_AM',
 'is_weekday',
 'subject_length',
 'subject_word_count',
 'outcome']
LABEL_COLUMN = 'outcome'
CATEGORICAL_COLUMNS = ["is_AM", "is_weekday"]
CONTINUOUS_COLUMNS = ['content_length',
 'content_word_count',
 'group_LeadingNetworking',
 'group_LeadingRestraining',
 'group_LeadingFacilitating',
 'group_LeadingMotivating',
 'group_LeadingPhysicalAction',
 'group_LeadingPraise',
 'group_LeadingCriticism',
 'group_ThinkingLearningOrCreative',
 'group_ThinkingSpiritual',
 'group_ThinkingNobility',
 'group_ThinkingAmbiguity',
 'group_ThinkingCurrentAffairs',
 'group_ThinkingAnalytical',
 'group_SpeakingFormality',
 'group_SpeakingPop',
 'group_SpeakingGeek',
 'group_SpeakingCasualAndFamily',
 'group_SpeakingMachismo',
 'group_SpeakingHumanity',
 'group_SpeakingDramatic',
 'group_SpeakingBanter',
 'group_ActingUrgency',
 'group_ActingIndustryJargon',
 'group_ActingOfficialeseAndLegalese',
 'group_ActingTechSpeak',
 'group_ActingProjectManagement',
 'group_SpeakingProfanity',
 'subject_length',
 'subject_word_count']

LABELS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]

def input_fn(df):
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values)
                     for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {k: tf.SparseTensor(
      indices=[[i, 0] for i in range(df[k].size)],
      values=df[k].values,
      dense_shape=[df[k].size, 1])
                      for k in CATEGORICAL_COLUMNS}
  # Merges the two dictionaries into one.
  feature_cols = dict(continuous_cols.items() + categorical_cols.items())

  # Converts the label column into a constant Tensor.
  label = tf.constant(df[LABEL_COLUMN].values)
  # Returns the feature columns and the label.
  return feature_cols, label

def train_input_fn():
  return input_fn(df_train)

def eval_input_fn():
  return input_fn(df_test)
# After reading in the data, you can train and evaluate the model:

classifier.fit(input_fn=train_input_fn, steps=200)
results = classifier.evaluate(input_fn=eval_input_fn, steps=1)
for key in sorted(results):
    print("%s: %s" % (key, results[key]))

y_pred = classifier.predict(input_fn=lambda: input_fn(df_val), as_iterable=False)
print(y_pred)

print('buckets found:',set(y_pred))

# # confusion matrix analysis
# from sklearn.metrics import confusion_matrix
# confusion_matrix(df_val[LABEL_COLUMN], y_pred)
# pd.crosstab(df_val[LABEL_COLUMN], y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)

# create some data
# https://stats.stackexchange.com/questions/95209/how-can-i-interpret-sklearn-confusion-matrix




























lookup = {0: 'LeadingNetworking', 1:'LeadingRestraining', 2:'LeadingFacilitating', 3:'LeadingMotivating', 4:'LeadingPhysicalAction', 5:'LeadingPraise', 6:'LeadingCriticism',
          7: 'ThinkingLearningOrCreative', 8:'ThinkingSpiritual', 9:'ThinkingNobility', 10:'ThinkingAmbiguity', 11:'ThinkingCurrentAffairs', 12:'ThinkingAnalytical', 13:'SpeakingFormality',
          14: 'SpeakingPop', 15:'SpeakingGeek', 16:'SpeakingCasualAndFamily', 17:'SpeakingMachismo', 18:'SpeakingHumanity', 19:'SpeakingDramatic', 20:'SpeakingBanter',
          21: 'ActingUrgency', 22:'ActingIndustryJargon', 23:'ActingOfficialeseAndLegalese', 24:'ActingTechSpeak', 25:'ActingProjectManagement', 26: 'SpeakingProfanity'}

y_truet = pd.Series([lookup[_] for _ in df_val[LABEL_COLUMN]])
y_predt = pd.Series([lookup[_] for _ in y_pred])
pd.crosstab(y_truet, y_predt, rownames=['Actual'], colnames=['Predicted'], margins=True)

subject_to_predict = "To the help desk"
content_to_predict = "fuck shit wank arse head"

def scrub_text(subject_to_predict, content_to_predict, categories):
  # prep text
  subject_to_predict = subject_to_predict.lower()
  pattern = re.compile('[^a-z]')
  subject_to_predict = re.sub(pattern, ' ', subject_to_predict)
  pattern = re.compile('\s+')
  subject_to_predict = re.sub(pattern, ' ', subject_to_predict)

  content_to_predict = content_to_predict.lower()
  pattern = re.compile('[^a-z]')
  content_to_predict = re.sub(pattern, ' ', content_to_predict)
  pattern = re.compile('\s+')
  content_to_predict = re.sub(pattern, ' ', content_to_predict)

  # get bag-of-words
  words_groups = []
  text = subject_to_predict + ' ' + content_to_predict
  for group_id in range(len(categories)):
    work_group = []
    print('Working bag number:', str(group_id))
    top_words = categories[group_id]
    work_group.append(len([w for w in text.split() if w in set(top_words)]))
    words_groups.append(work_group)

  # count emails per category group and feature engineering
  raw_text = []
  subject_length = []
  subject_word_count = []
  content_length = []
  content_word_count = []
  is_am_list = []
  is_weekday_list = []
  group_LeadingNetworking = []
  group_LeadingRestraining = []
  group_LeadingFacilitating = []
  group_LeadingMotivating = []
  group_LeadingPhysicalAction = []
  group_LeadingPraise = []
  group_LeadingCriticism = []
  group_ThinkingLearningOrCreative = []
  group_ThinkingSpiritual = []
  group_ThinkingNobility = []
  group_ThinkingAmbiguity = []
  group_ThinkingCurrentAffairs = []
  group_ThinkingAnalytical = []
  group_SpeakingFormality = []
  group_SpeakingPop = []
  group_SpeakingGeek = []
  group_SpeakingCasualAndFamily = []
  group_SpeakingMachismo = []
  group_SpeakingHumanity = []
  group_SpeakingDramatic = []
  group_SpeakingBanter = []
  group_ActingUrgency = []
  group_ActingIndustryJargon = []
  group_ActingOfficialeseAndLegalese = []
  group_ActingTechSpeak = []
  group_ActingProjectManagement = []
  group_SpeakingProfanity = []
  group_Praise = []
  group_Criticism = []
  final_outcome = []

  cur_time_stamp = datetime.datetime.now()

  raw_text.append(text)
  group_LeadingNetworking.append(words_groups[0])
  group_LeadingRestraining.append(words_groups[1])
  group_LeadingFacilitating.append(words_groups[2])
  group_LeadingMotivating.append(words_groups[3])
  group_LeadingPhysicalAction.append(words_groups[4])
  group_LeadingPraise.append(words_groups[5])
  group_LeadingCriticism.append(words_groups[6])
  group_ThinkingLearningOrCreative.append(words_groups[7])
  group_ThinkingSpiritual.append(words_groups[8])
  group_ThinkingNobility.append(words_groups[9])
  group_ThinkingAmbiguity.append(words_groups[10])
  group_ThinkingCurrentAffairs.append(words_groups[11])
  group_ThinkingAnalytical.append(words_groups[12])
  group_SpeakingFormality.append(words_groups[13])
  group_SpeakingPop.append(words_groups[14])
  group_SpeakingGeek.append(words_groups[15])
  group_SpeakingCasualAndFamily.append(words_groups[16])
  group_SpeakingMachismo.append(words_groups[17])
  group_SpeakingHumanity.append(words_groups[18])
  group_SpeakingDramatic.append(words_groups[19])
  group_SpeakingBanter.append(words_groups[20])
  group_ActingUrgency.append(words_groups[21])
  group_ActingIndustryJargon.append(words_groups[22])
  group_ActingOfficialeseAndLegalese.append(words_groups[23])
  group_ActingTechSpeak.append(words_groups[24])
  group_ActingProjectManagement.append(words_groups[25])
  group_SpeakingProfanity.append(words_groups[26])
  outcome_tots = [words_groups[0], words_groups[1], words_groups[2], words_groups[3], words_groups[4], words_groups[5], words_groups[6],
  words_groups[7], words_groups[8], words_groups[9], words_groups[10], words_groups[11], words_groups[12], words_groups[13],
  words_groups[14], words_groups[15], words_groups[16], words_groups[17], words_groups[18], words_groups[19], words_groups[20], words_groups[21],
  words_groups[22], words_groups[23], words_groups[24], words_groups[25], words_groups[26]]
  final_outcome.append(outcome_tots.index(max(outcome_tots)))

  subject_length.append(len(subject_to_predict))
  subject_word_count.append(len(subject_to_predict.split()))
  content_length.append(len(content_to_predict))
  content_word_count.append(len(content_to_predict.split()))
  dt = cur_time_stamp
  is_am = 'no'
  if (dt.time() < datetime.time(12)): is_am = 'yes'
  is_am_list.append(is_am)
  is_weekday = 'no'
  if (dt.weekday() < 6): is_weekday = 'yes'
  is_weekday_list.append(is_weekday)

  # add simple engineered features?
  training_set = pd.DataFrame({
                "raw_text":raw_text,
                "group_LeadingNetworking":group_LeadingNetworking[0],
                "group_LeadingRestraining":group_LeadingRestraining[0],
                "group_LeadingFacilitating":group_LeadingFacilitating[0],
                "group_LeadingMotivating":group_LeadingMotivating[0],
                "group_LeadingPhysicalAction":group_LeadingPhysicalAction[0],
                "group_LeadingPraise":group_LeadingPraise[0],
                "group_LeadingCriticism":group_LeadingCriticism[0],
                "group_ThinkingLearningOrCreative":group_ThinkingLearningOrCreative[0],
                "group_ThinkingSpiritual":group_ThinkingSpiritual[0],
                "group_ThinkingNobility":group_ThinkingNobility[0],
                "group_ThinkingAmbiguity":group_ThinkingAmbiguity[0],
                "group_ThinkingCurrentAffairs":group_ThinkingCurrentAffairs[0],
                "group_ThinkingAnalytical":group_ThinkingAnalytical[0],
                "group_SpeakingFormality":group_SpeakingFormality[0],
                "group_SpeakingPop":group_SpeakingPop[0],
                "group_SpeakingGeek":group_SpeakingGeek[0],
                "group_SpeakingCasualAndFamily":group_SpeakingCasualAndFamily[0],
                "group_SpeakingMachismo":group_SpeakingMachismo[0],
                "group_SpeakingHumanity":group_SpeakingHumanity[0],
                "group_SpeakingDramatic":group_SpeakingDramatic[0],
                "group_SpeakingBanter":group_SpeakingBanter[0],
                "group_ActingUrgency":group_ActingUrgency[0],
                "group_ActingIndustryJargon":group_ActingIndustryJargon[0],
                "group_ActingOfficialeseAndLegalese":group_ActingOfficialeseAndLegalese[0],
                "group_ActingTechSpeak":group_ActingTechSpeak[0],
                "group_ActingProjectManagement":group_ActingProjectManagement[0],
                "group_SpeakingProfanity":group_SpeakingProfanity[0],
                "subject_length":subject_length,
                "subject_word_count":subject_word_count,
                "content_length":content_length,
                "content_word_count":content_word_count,
                "is_AM":is_am_list,
                "is_weekday":is_weekday_list,
                "outcome":final_outcome})


  return(training_set)

scrubbed_entry = scrub_text(subject_to_predict, content_to_predict, categories)

y_pred = classifier.predict(input_fn=lambda: input_fn(scrubbed_entry), as_iterable=False)
print(y_pred)

category_names = [ 'LeadingNetworking',
 'LeadingRestraining',
 'LeadingFacilitating',
 'LeadingMotivating',
 'LeadingPhysicalAction',
 'LeadingPraise',
 'LeadingCriticism',
 'ThinkingLearningOrCreative',
 'ThinkingSpiritual',
 'ThinkingNobility',
 'ThinkingAmbiguity',
 'ThinkingCurrentAffairs',
 'ThinkingAnalytical',
 'SpeakingFormality',
 'SpeakingPop',
 'SpeakingGeek',
 'SpeakingCasualAndFamily',
 'SpeakingMachismo',
 'SpeakingHumanity',
 'SpeakingDramatic',
 'SpeakingBanter',
 'ActingUrgency',
 'ActingIndustryJargon',
 'ActingOfficialeseAndLegalese',
 'ActingTechSpeak',
 'ActingProjectManagement',
 'SpeakingProfanity']

print('Forward request to: ' +  category_names[y_pred[0]])
