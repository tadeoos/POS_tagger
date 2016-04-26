# Słownik mapujący domyślne tagi korpusu Brown na 65 tagów; by TT

map_dict = dict([['*', '*'],["STOP", "STOP"],["'",  "."],["''", "."],['(', "."],['(-HL', "."],[')', "."],[')-HL', "."],['*-HL', "."],['*-NC', "."],['*-TL', "."],[',', "."],[',-HL', "."],[',-NC', "."],[',-TL', "."],['--', "."],['---HL', "."],['.', "."],['.-HL', "."],['.-NC', "."],['.-TL', "."],[':', "."],[':-HL', "."],[':-TL', "."],['ABL', "AB"],['ABN', "AB"],['ABN-HL', "AB"],['ABN-NC', "AB"],['ABN-TL', "AB"],['ABX', "AB"],['AP',  "AP"],['AP$', "AP"],['AP+AP-NC', "AP"],['AP-HL', "AP"],['AP-NC', "AP"],['AP-TL', "AP"],['AT', 'AT'],['AT-HL', 'AT'],['AT-NC', 'AT'],['AT-TL', 'AT'],['AT-TL-HL', 'AT'],['BE', 'BE'],['BE-HL', 'BE'],['BE-TL', 'BE'],['BED', 'BE'],['BED*', 'BE'],['BED-NC', 'BE'],['BEDZ', 'BE'],['BEDZ*', 'BE'],['BEDZ-HL', 'BE'],['BEDZ-NC', 'BE'],['BEG', 'BE'],['BEM', 'BE'],['BEM*', 'BE'],['BEM-NC', 'BE'],['BEN', 'BE'],['BEN-TL', 'BE'],['BER', 'BE'],['BER*', 'BE'],['BER*-NC', 'BE'],['BER-HL', 'BE'],['BER-NC', 'BE'],['BER-TL', 'BE'],['BEZ', 'BE'],['BEZ*', 'BE'],['BEZ-HL', 'BE'],['BEZ-NC', 'BE'],['BEZ-TL', 'BE'],['CC', 'CC'],['CC-HL', 'CC'],['CC-NC', 'CC'],['CC-TL', 'CC'],['CC-TL-HL', 'CC'],['CD', 'CD'],['CD$', 'CD'],['CD-HL', 'CD'],['CD-NC', 'CD'],['CD-TL', 'CD'],['CD-TL-HL', 'CD'],['CS', 'CS'],['CS-HL', 'CS'],['CS-NC', 'CS'],['CS-TL', 'CS'],['DO', 'DO'],['DO*', 'DO+'],['DO*-HL', 'DO+'],['DO+PPSS', 'DO+'],['DO-HL', 'DO+'],['DO-NC', 'DO+'],['DO-TL', 'DO+'],['DOD', 'DO+'],['DOD*', 'DO+'],['DOD*-TL', 'DO+'],['DOD-NC', 'DO+'],['DOZ', 'DO+'],['DOZ*', 'DO+'],['DOZ*-TL', 'DO+'],['DOZ-HL', 'DO+'],['DOZ-TL', 'DO+'],['DT', 'DT'],['DT$', 'DT+'],['DT+BEZ', 'DT+'],['DT+BEZ-NC', 'DT+'],['DT+MD', 'DT+'],['DT-HL', 'DT+'],['DT-NC', 'DT+'],['DT-TL', 'DT+'],['DTI', 'DT+'],['DTI-HL', 'DT+'],['DTI-TL', 'DT+'],['DTS', 'DT+'],['DTS+BEZ', 'DT+'],['DTS-HL', 'DT+'],['DTX', 'DT+'],['EX', 'EX'],['EX+BEZ', 'EX'],['EX+HVD', 'EX'],['EX+HVZ', 'EX'],['EX+MD', 'EX'],['EX-HL', 'EX'],['EX-NC', 'EX'],['FW-*',  'FW'],['FW-*-TL', 'FW'],['FW-AT', 'FW'],['FW-AT+NN-TL', 'FW'],['FW-AT+NP-TL', 'FW'],['FW-AT-HL', 'FW'],['FW-AT-TL', 'FW'],['FW-BE', 'FW'],['FW-BER', 'FW'],['FW-BEZ', 'FW'],['FW-CC', 'FW'],['FW-CC-TL', 'FW'],['FW-CD', 'FW'],['FW-CD-TL', 'FW'],['FW-CS', 'FW'],['FW-DT', 'FW'],['FW-DT+BEZ', 'FW'],['FW-DTS', 'FW'],['FW-HV', 'FW'],['FW-IN', 'FW'],['FW-IN+AT', 'FW'],['FW-IN+AT-T', 'FW'],['FW-IN+AT-TL', 'FW'],['FW-IN+NN', 'FW'],['FW-IN+NN-TL', 'FW'],['FW-IN+NP-TL', 'FW'],['FW-IN-TL', 'FW'],['FW-JJ', 'FW'],['FW-JJ-NC', 'FW'],['FW-JJ-TL', 'FW'],['FW-JJR', 'FW'],['FW-JJT', 'FW'],['FW-NN', 'FW'],['FW-NN$', 'FW'],['FW-NN$-TL', 'FW'],['FW-NN-NC', 'FW'],['FW-NN-TL', 'FW'],['FW-NN-TL-NC', 'FW'],['FW-NNS', 'FW'],['FW-NNS-NC', 'FW'],['FW-NNS-TL', 'FW'],['FW-NP', 'FW'],['FW-NP-TL', 'FW'],['FW-NPS', 'FW'],['FW-NPS-TL', 'FW'],['FW-NR', 'FW'],['FW-NR-TL', 'FW'],['FW-OD-NC', 'FW'],['FW-OD-TL', 'FW'],['FW-PN', 'FW'],['FW-PP$', 'FW'],['FW-PP$-NC', 'FW'],['FW-PP$-TL', 'FW'],['FW-PPL', 'FW'],['FW-PPL+VBZ', 'FW'],['FW-PPO', 'FW'],['FW-PPO+IN', 'FW'],['FW-PPS', 'FW'],['FW-PPSS', 'FW'],['FW-PPSS+HV', 'FW'],['FW-QL', 'FW'],['FW-RB', 'FW'],['FW-RB+CC', 'FW'],['FW-RB-TL', 'FW'],['FW-TO+VB', 'FW'],['FW-UH', 'FW'],['FW-UH-NC', 'FW'],['FW-UH-TL', 'FW'],['FW-VB', 'FW'],['FW-VB-NC', 'FW'],['FW-VB-TL', 'FW'],['FW-VBD', 'FW'],['FW-VBD-TL', 'FW'],['FW-VBG', 'FW'],['FW-VBG-TL', 'FW'],['FW-VBN', 'FW'],['FW-VBZ', 'FW'],['FW-WDT', 'FW'],['FW-WPO', 'FW'],['FW-WPS', 'FW'],['HV', 'HV'],['HV*', 'HV+'],['HV+TO', 'HV+'],['HV-HL', 'HV+'],['HV-NC', 'HV+'],['HV-TL', 'HV+'],['HVD', 'HV+'],['HVD*', 'HV+'],['HVD-HL', 'HV+'],['HVG', 'HV+'],['HVG-HL', 'HV+'],['HVN', 'HV+'],['HVZ', 'HV+'],['HVZ*', 'HV+'],['HVZ-NC', 'HV+'],['HVZ-TL', 'HV+'],['IN', 'IN'],['IN+IN', 'IN+'],['IN+PPO', 'IN+'],['IN-HL', 'IN+'],['IN-NC', 'IN+'],['IN-TL', 'IN+'],['IN-TL-HL', 'IN+'],['JJ', 'JJ'],['JJ$-TL', 'JJ+'],['JJ+JJ-NC', 'JJ+'],['JJ-HL', 'JJ+'],['JJ-NC', 'JJ+'],['JJ-TL', 'JJ+'],['JJ-TL-HL', 'JJ+'],['JJ-TL-NC', 'JJ+'],['JJR', 'JJ+'],['JJR+CS', 'JJ+'],['JJR-HL', 'JJ+'],['JJR-NC', 'JJ+'],['JJR-TL', 'JJ+'],['JJS', 'JJ+'],['JJS-HL', 'JJ+'],['JJS-TL', 'JJ+'],['JJT', 'JJ+'],['JJT-HL', 'JJ+'],['JJT-NC', 'JJ+'],['JJT-TL', 'JJ+'],['MD', 'MD'],['MD*', 'MD+'],['MD*-HL', 'MD+'],['MD+HV', 'MD+'],['MD+PPSS', 'MD+'],['MD+TO', 'MD+'],['MD-HL', 'MD+'],['MD-NC', 'MD+'],['MD-TL', 'MD+'],['NIL', 'NIL'],['NN', 'NN'],['NN$', 'NN+'],['NN$-HL', 'NN+'],['NN$-TL', 'NN+'],['NN+BEZ', 'NN+'],['NN+BEZ-TL', 'NN+'],['NN+HVD-TL', 'NN+'],['NN+HVZ', 'NN+'],['NN+HVZ-TL', 'NN+'],['NN+IN', 'NN+'],['NN+MD', 'NN+'],['NN+NN-NC', 'NN+'],['NN-HL', 'NN+'],['NN-NC', 'NN+'],['NN-TL', 'NN+'],['NN-TL-HL', 'NN+'],['NN-TL-NC', 'NN+'],['NNS', 'NNS'],['NNS$', 'NNS+'],['NNS$-HL', 'NNS+'],['NNS$-NC', 'NNS+'],['NNS$-TL', 'NNS+'],['NNS$-TL-HL', 'NNS+'],['NNS+MD', 'NNS+'],['NNS-HL', 'NNS+'],['NNS-NC', 'NNS+'],['NNS-TL', 'NNS+'],['NNS-TL-HL', 'NNS+'],['NNS-TL-NC', 'NNS+'],['NP', 'NP'], ['NP$', 'NP+'],['NP$-HL', 'NP+'],['NP$-TL', 'NP+'],['NP+BEZ', 'NP+'],['NP+BEZ-NC', 'NP+'],['NP+HVZ', 'NP+'],['NP+HVZ-NC', 'NP+'],['NP+MD', 'NP+'],['NP-HL', 'NP+'],['NP-NC', 'NP+'],['NP-TL', 'NP+'],['NP-TL-HL', 'NP+'],['NPS', 'NPS'],['NPS$', 'NPS+'],['NPS$-HL', 'NPS+'],['NPS$-TL', 'NPS+'],['NPS-HL', 'NPS+'],['NPS-NC', 'NPS+'],['NPS-TL', 'NPS+'],['NR', 'NR'],['NR$', 'NR+'],['NR$-TL', 'NR+'],['NR+MD', 'NR+'],['NR-HL', 'NR+'],['NR-NC', 'NR+'],['NR-TL', 'NR+'],['NR-TL-HL', 'NR+'],['NRS', 'NR+'],['NRS-TL', 'NR+'],['OD', 'OD'],['OD-HL', 'OD'],['OD-NC', 'OD'],['OD-TL', 'OD'],['PN', 'PN'],['PN$', 'PN+'],['PN+BEZ', 'PN+'],['PN+HVD', 'PN+'],['PN+HVZ', 'PN+'],['PN+MD', 'PN+'],['PN-HL', 'PN+'],['PN-NC', 'PN+'],['PN-TL', 'PN+'],['PP$', 'PP$'],['PP$$', 'PP$'],['PP$-HL', 'PP$'],['PP$-NC', 'PP$'],['PP$-TL', 'PP$'],['PPL', 'PPL'],['PPL-HL', 'PPL'],['PPL-NC', 'PPL'],['PPL-TL', 'PPL'],['PPLS', 'PPL'],['PPO', 'PPO'],['PPO-HL', 'PPO'],['PPO-NC', 'PPO'],['PPO-TL', 'PPO'],['PPS', 'PPS'],['PPS+BEZ', 'PPS+'],['PPS+BEZ-HL', 'PPS+'],['PPS+BEZ-NC', 'PPS+'],['PPS+HVD', 'PPS+'],['PPS+HVZ', 'PPS+'],['PPS+MD', 'PPS+'],['PPS-HL', 'PPS+'],['PPS-NC', 'PPS+'],['PPS-TL', 'PPS+'],['PPSS', 'PPSS'],['PPSS+BEM', 'PPSS+'],['PPSS+BER', 'PPSS+'],['PPSS+BER-N', 'PPSS+'],['PPSS+BER-NC', 'PPSS+'],['PPSS+BER-TL', 'PPSS+'],['PPSS+BEZ', 'PPSS+'],['PPSS+BEZ*', 'PPSS+'],['PPSS+HV', 'PPSS+'],['PPSS+HV-TL', 'PPSS+'],['PPSS+HVD', 'PPSS+'],['PPSS+MD', 'PPSS+'],['PPSS+MD-NC', 'PPSS+'],['PPSS+VB', 'PPSS+'],['PPSS-HL', 'PPSS+'],['PPSS-NC', 'PPSS+'],['PPSS-TL', 'PPSS+'],['QL', 'QL'],['QL-HL', 'QL'],['QL-NC', 'QL'],['QL-TL', 'QL'],['QLP', 'QL'],['RB', 'RB'],['RB$', 'RB+'],['RB+BEZ', 'RB+'],['RB+BEZ-HL', 'RB+'],['RB+BEZ-NC', 'RB+'],['RB+CS', 'RB+'],['RB-HL', 'RB+'],['RB-NC', 'RB+'],['RB-TL', 'RB+'],['RBR', 'RB+'],['RBR+CS', 'RB+'],['RBR-NC', 'RB+'],['RBT',  'RB+'],['RN', 'RN'],['RP', 'RP'],['RP+IN', 'RP'],['RP-HL', 'RP'],['RP-NC', 'RP'],['RP-TL', 'RP'],['TO', 'TO'],['TO+VB', 'TO'],['TO-HL', 'TO'],['TO-NC', 'TO'],['TO-TL', 'TO'],['UH', 'UH'],['UH-HL', 'UH'],['UH-NC', 'UH'],['UH-TL', 'UH'],['VB', 'VB'],['VB+AT', 'VB'],['VB+IN', 'VB'],['VB+JJ-NC', 'VB'],['VB+PPO', 'VB'],['VB+RP', 'VB'],['VB+TO', 'VB'],['VB+VB-NC', 'VB'],['VB-HL', 'VB'],['VB-NC', 'VB'],['VB-TL', 'VB'],['VBD', 'VBD'],['VBD-HL', 'VBD'],['VBD-NC', 'VBD'],['VBD-TL', 'VBD'],['VBG', 'VBG'],['VBG+TO', 'VBG'],['VBG-HL', 'VBG'],['VBG-NC', 'VBG'],['VBG-TL', 'VBG'],['VBN', 'VBN'],['VBN+TO', 'VBN'],['VBN-HL', 'VBN'],['VBN-NC', 'VBN'],['VBN-TL', 'VBN'],['VBN-TL-HL', 'VBN'],['VBN-TL-NC', 'VBN'],['VBZ', 'VBZ'],['VBZ-HL', 'VBZ'],['VBZ-NC', 'VBZ'],['VBZ-TL', 'VBZ'],['WDT', 'WDT'],['WDT+BER', 'WDT+'],['WDT+BER+PP', 'WDT+'],['WDT+BEZ', 'WDT+'],['WDT+BEZ-HL', 'WDT+'],['WDT+BEZ-NC', 'WDT+'],['WDT+BEZ-TL', 'WDT+'],['WDT+DO+PPS', 'WDT+'],['WDT+DOD', 'WDT+'],['WDT+HVZ', 'WDT+'],['WDT-HL', 'WDT+'],['WDT-NC', 'WDT+'],['WP$', 'WP$'],['WPO', 'WPO'],['WPO-NC', 'WPO'],['WPO-TL', 'WPO'],['WPS', 'WPS'],['WPS+BEZ', 'WPS'],['WPS+BEZ-NC', 'WPS'],['WPS+BEZ-TL', 'WPS'],['WPS+HVD', 'WPS'],['WPS+HVZ', 'WPS'],['WPS+MD', 'WPS'],['WPS-HL', 'WPS'],['WPS-NC', 'WPS'],['WPS-TL', 'WPS'],['WQL', 'WQL'],['WQL-TL', 'WQL'],['WRB', 'WRB'],['WRB+BER', 'WRB+'],['WRB+BEZ', 'WRB+'],['WRB+BEZ-TL', 'WRB+'],['WRB+DO', 'WRB+'],['WRB+DOD', 'WRB+'],['WRB+DOD*', 'WRB+'],['WRB+DOZ', 'WRB+'],['WRB+IN', 'WRB+'],['WRB+MD', 'WRB+'],['WRB-HL', 'WRB+'],['WRB-NC', 'WRB+'],['WRB-TL', 'WRB+'],['``', '``']])