import rdflib
import sys
import os
import re
import json
import torch
import numpy as np
import random
import pickle

RDFNameSpace = rdflib.Namespace("http://ANYBURL.edu/")
RDFNameSpace.node = rdflib.term.URIRef("http://ANYBURL.edu/node/")
RDFNameSpace.relation = rdflib.term.URIRef("http://ANYBURL.edu/relation/")
RDFNameSpace.CHARACTERS_TO_REMOVE = ["<", ">"]

class ANYBURLScorer:
    
    __slots__ = "rdfGraph", "generatedTriples", "relationIdToNameMap", "parsedRules", "model"

    outerlist = []

    def __init__(self, folder, dataset=None):

        self.model = ANYBURLScorer.dummy()
        self.model.load_checkpoint = self.load_checkpoint

        if dataset is None:
            self.rdfGraph = None
            self.generatedTriples = None
            self.relationIdToNameMap = None
            self.parsedRules = None
            return

        dataset_name = ""
        if dataset==0:
            dataset_name="FB13"
        if dataset==1:
            dataset_name="FB15K"
        if dataset==2:
            dataset_name="FB15K237"
        if dataset==3:
            dataset_name="NELL-995"
        if dataset==4:
            dataset_name="WN11"
        if dataset==5:
            dataset_name="WN18"
        if dataset==6:
            dataset_name="WN18RR"
        if dataset==7:
            dataset_name="YAGO3-10"

        datasetPath =  folder + "Datasets/" + dataset_name + "/" 
        data_file_names = ["new_train2id.txt", "new_valid2id.txt"]
        KBPath = []
        for filename in data_file_names:
            KBPath.append(datasetPath + filename)

        rulesPath = folder + "Datasets/" + dataset_name + "/ANYBURLResults/rules.txt"
        entity2idFilePath = "Datasets/" + dataset_name + "/entity2id.txt"
        relation2idFilePath = "Datasets/" + dataset_name + "/relation2id.txt"
        
        # Read through the id files
        entityNameToidMap,entityIdToNameMap, = self.readIdFiles(entity2idFilePath)
        relationNameToIdMap, relationIdToNameMap = self.readIdFiles(relation2idFilePath)
        
        # Load the training triples in the KBase
        # print("Loading data into RDF Database ...")
        rdfGraph = self.load_ANYBURL(KBPath)

        print("Data Loading completed. Parsing ANYBURL results file ...")

        # parse results from ANYBURL execution
        parsedRules = self.parseANYBURLResults(rulesPath)
        print(parsedRules["parents"][0])

        print("Parsed AnyBurl rules. Extracting triples based on parsed rules ...")
        
        # # list of relations for which rules new rules have generated
        parsedRuleRelations = list(parsedRules.keys())
        
        # # generate facts for each of the new rules of the relations
        generatedTriples = dict()
        count = 0
        for relation in parsedRuleRelations:
            count +=1
            print("generating facts of relation:", relation, "Progress: ", count , " of ",  len(parsedRuleRelations))
            generatedTriples[relation] = self.generateTriplesOfRelation(parsedRules, relation , relationNameToIdMap, rdfGraph)
        print("Triple parsing completed.")

        self.rdfGraph = rdfGraph
        self.generatedTriples = generatedTriples
        self.relationIdToNameMap = relationIdToNameMap
        self.parsedRules = parsedRules
    
    def predict(self, values):
        batch_h = values["batch_h"]
        batch_r = values["batch_r"]
        batch_t = values["batch_t"]
        mode = values["mode"]

        # Assuming batch_ are all pytorch 1-D Variables
        triplesCount = list(batch_h.size())[0]
        scores = list()

        for i in range(triplesCount):
            head = int(batch_h[i].item())
            relation = int(batch_r[i].item())
            tail = int(batch_t[i].item())
            scores.append(self.computeTripleScore(head, relation, tail))
        return scores

    def computeTripleScore(self, head, relation, tail):
        if(self.checkIfTripleInKB(self.rdfGraph, head, relation, tail)):
            return 0 ## best score possible
        else:
            return self.checkIfTripleWasGenerated(self.generatedTriples, head, relation, tail, self.relationIdToNameMap)

    def load_ANYBURL(self, KBPath):

        fileTriples = list()
        for datafile in KBPath:
            fileLines = open(datafile).readlines()
            for line in fileLines[1:]:
                head, tail, relation = line.split()
                head, tail, relation = head.strip(), tail.strip(), relation.strip()
                fileTriples.append([head, relation, tail])
        rdfGraph = rdflib.Graph()

        rdfGraph.bind("node", RDFNameSpace.node)
        rdfGraph.bind("relation", RDFNameSpace.relation)

        for triple in fileTriples:
            rdfGraph.add( ( RDFNameSpace.node + triple[0], RDFNameSpace.relation + triple[1],  RDFNameSpace.node + triple[2]) )
        return rdfGraph

    def readIdFiles(self, filename):
        filelines = open(filename, "r").readlines()
        nameToId = dict()
        idToName = dict()
        for line in filelines[1:]:
            element_name, element_id = line.split("\t")
            element_name, element_id = element_name.strip(), element_id.strip()
            nameToId[element_name] = element_id
            idToName[element_id] = element_name
        return nameToId, idToName

    def getRDFMatches(self, rdfGraph, unknown=None, head=None, tail=None, relation=None):

        if unknown == "head":
            return rdfGraph.subjects(RDFNameSpace.relation + relation, RDFNameSpace.node + tail)
        elif unknown == "tail":
            return rdfGraph.objects(RDFNameSpace.node + head, RDFNameSpace.relation + relation)
        elif unknown == "relation":
            return rdfGraph.predicates(RDFNameSpace.node + head, RDFNameSpace.node + tail)

    def executeRDFQuery(self, rdfGraph, query):
        return rdfGraph.query(query)

    def parseTripleString(self, tripleString):
        #['<place_of_death>(X,<tripoli_lebanon>)']
        relation = tripleString.split(">")[0]
        headTailList = tripleString.split(">")[1:]
        headTail = ""
        for element in headTailList:
            headTail += str(element)

        headTail.replace("(", "")
        headTail.replace(")", "")
        head , tail = headTail.split(",")
        head = head.replace("(", "")
        tail = tail.replace(")", "")
        
        if "<" in head:
            head = head + ">"

        if "<" in tail:
            tail = tail + ">"

        relation = relation.replace("<", "")
        return {"head":head, "relation": relation, "tail": tail}


    def parseGeneratedRules(self, generatedRules):
        parsedRules = dict()
        for rule in generatedRules:
            concatenatedRuleHead, concatenatedRuleBody = rule.split("<=")
            concatenatedRuleHead, concatenatedRuleBody = concatenatedRuleHead.strip(), concatenatedRuleBody.strip()
            splitRuleBody = concatenatedRuleBody.split()

            splitRuleHead = concatenatedRuleHead.split()
            rulePossibleCount = int(splitRuleHead[0])
            ruleSupport = int(splitRuleHead[1])
            ruleCoverage = float(splitRuleHead[2])
            ruleHead = self.parseTripleString(splitRuleHead[3])
            ruleRelation = ruleHead["relation"]

            ruleBody = list()

            for rule in concatenatedRuleBody.split("),"):
                ruleBody.append(self.parseTripleString(rule))
                        
            ruleSummary = dict()
            ruleSummary["head"] = ruleHead
            ruleSummary["body"] = ruleBody
            ruleSummary["support"] = ruleSupport
            ruleSummary["coverage"] = ruleCoverage
            ruleSummary["total"] = rulePossibleCount
            
            if ruleRelation not in parsedRules:
                parsedRules[ruleRelation] = list()
            
            parsedRules[ruleRelation].append(ruleSummary)
        
        # sorting rules based on coverage 
        for key in parsedRules:
            current_value = parsedRules[key]
            current_value = sorted(current_value, key = lambda x: x["coverage"], reverse=True)
            parsedRules[key] = current_value
        return parsedRules

    def parseANYBURLResults(self, rulesPath):
        fileData = open(rulesPath, "r").readlines()
        return self.parseGeneratedRules(fileData)

    def generateTriplesOfRelation(self, parsedRules, relationName, relationNameToidMap, rdfGraph):
        results, queries, confidence = list(), list(), list()
        generatedTriples = set()
        
        if relationName not in parsedRules:
            return None

        for i in range(len(parsedRules[relationName])):
            try:
                head = parsedRules[relationName][i]["head"]
                body = parsedRules[relationName][i]["body"]

                # query start and add head elements
                query = "SELECT  "
                rule_head_head, rule_head_tail = str(head["head"]), str(head["tail"])
                if "<" in rule_head_head:
                    rule_head_head = rule_head_head.replace("<", "")
                    rule_head_head = rule_head_head.replace(">", "")
                    query += "(str(node:" + rule_head_head + ") as ?label) "
                else:
                    query += "?"+rule_head_head + " "

                if "<" in rule_head_tail:
                    rule_head_tail = rule_head_tail.replace("<", "")
                    rule_head_tail = rule_head_tail.replace(">", "")
                    query += "(str(node:" + rule_head_tail + ") as ?label) WHERE {"
                else:
                    query += "?" + rule_head_tail + " WHERE {"
                # add body elements
                for j in range(len(body)):
                    rule_body_head, rule_body_relation, rule_body_tail = str(body[j]["head"]), str(body[j]["relation"]), str(body[j]["tail"]) 

                    if "<" in rule_body_head:
                        rule_body_head = rule_body_head.replace("<", "")
                        rule_body_head = rule_body_head.replace(">", "")
                        query += " node:" + rule_body_head + " "
                    else:
                        query +=  " ?" + rule_body_head + " "
                    query += " relation:" + str(relationNameToidMap[rule_body_relation]).strip() + " "
                    if "<" in rule_body_tail:
                        rule_body_tail =  rule_body_tail.replace("<", "")
                        rule_body_tail =  rule_body_tail.replace(">", "")
                        query += "node:" + rule_body_tail + " ."
                    else:
                        query += "?" + rule_body_tail + " ."
                # add closing tag
                query = query[:-1] + "}"
                
                queries.append(query)
                confidence.append(parsedRules[relationName][i]["coverage"] )

            except:
                continue
        
        if rdfGraph:
            tripleConfidenceMap = dict()
            relationTriples = set()

            for i in range(len(queries)):
                query = queries[i]
                print(query, " ", i+1 , " of", len(queries) )
                result = rdfGraph.query(query)
                temp_res = set()
                for s, p in result:
                    triple = str(s) + "," + str(p)
                    if triple in tripleConfidenceMap:
                        if confidence[i] > tripleConfidenceMap[triple]:
                            tripleConfidenceMap[triple] = confidence[i]
                            temp_res.add(str(s) + "," + str(p))
                    else:
                        tripleConfidenceMap[triple] = confidence[i]
                        temp_res.add(str(s) + "," + str(p))
                if temp_res:
                    results.append([query, temp_res, confidence[i]])

        # sort triples based on the confidence values
        results.sort(key = lambda x: float(x[2]), reverse = True)
        
        return results

    def checkIfTripleInKB(self, rdfGraph, head, relation, tail):
        head = RDFNameSpace.node + str(head)
        relation = RDFNameSpace.relation + str(relation)
        tail = RDFNameSpace.node + str(tail)
        if (head, relation, tail) in rdfGraph:
            return True
        return False

    def checkIfTripleWasGenerated(self, generatedTriples, head, relation, tail, relationIdToNameMap):
        relation = str(relation)
        relationName = relationIdToNameMap[relation]

        # additional triples were not generated for the rule
        if relationName not in generatedTriples:
            return 1
        
        head = RDFNameSpace.node + str(head)
        tail = RDFNameSpace.node + str(tail)
        tripleEntry = str(head) + "," + str(tail)
        relationTriplesDataList = generatedTriples[relationName]
        
        for triplesData in relationTriplesDataList:
            smallRandomValue = random.randint(0,1000) * np.nextafter(0, 1) # np.nextafter returns ~ 5e-324 , multiplying even 1000 is really small
            query = triplesData[0]
            triples = triplesData[1]
            confidence = float(triplesData[2]) + smallRandomValue
            if tripleEntry in triples:
                return 1 - confidence
        return 1

    def load_checkpoint(self, filename):
        with open(filename, "rb") as f:
            model = pickle.load(f)
            self.model = model
            self.model.model = model

    def saveModel(self, filename, model):
        with open(filename, "wb") as f:
            pickle.dump(model, f) 
        print("Model storing complete")


    class dummy():
            def __init__(self):
                return

def main():
    folder = sys.argv[1]
    dataset = int(sys.argv[2])
    ANYBURLModelName = folder + "Model/" +str(dataset) + "/anyburl.model"
    
    values = dict()
    values["batch_h"] = torch.Tensor([3, 3, 0, 3, 1111])
    values["batch_r"] = torch.Tensor([10, 10, 1, 10, 1])
    values["batch_t"] = torch.Tensor([3, 3, 0, 46098, 1111])
    values["mode"] = "normal"

    model = ANYBURLScorer(folder, dataset)
    
    model.saveModel(ANYBURLModelName, model)

    # load an existing model to a variable
    # model = ANYBURLScorer()
    # model.model.load_checkpoint(ANYBURLModelName)
    # print(model.model.predict(values))
    # print(model.model.generatedTriples)

if __name__ == "__main__":
    main()
