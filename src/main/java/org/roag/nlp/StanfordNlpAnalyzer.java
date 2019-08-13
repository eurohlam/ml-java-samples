package org.roag.nlp;

import edu.stanford.nlp.ie.util.RelationTriple;
import edu.stanford.nlp.pipeline.*;

import java.util.*;
import java.util.stream.Collectors;
import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.LogManager;

/**
 * Created by eurohlam on 7/08/2019.
 * Some more deep experiments with Stanford CoreNLP API
 */
public class StanfordNlpAnalyzer {

    private static final Logger LOG = LogManager.getLogger(StanfordNlpAnalyzer.class);

    private static final String text = "Marley was dead: to begin with. There is no doubt whatever about that. " +
            "The register of his burial was signed by the clergyman, the clerk, the undertaker, and the chief mourner. " +
            "Scrooge signed it: and Scrooge’s name was good upon ’Change, for anything he chose to put his hand to. " +
            "Old Marley was as dead as a door-nail. " +
            "Mind! I don’t mean to say that I know, of my own knowledge, what there is particularly dead about a door-nail. " +
            "I might have been inclined, myself, to regard a coffin-nail as the deadest piece of ironmongery in the trade. " +
            "But the wisdom of our ancestors is in the simile; and my unhallowed hands shall not disturb it, or the Country’s done for. " +
            "You will therefore permit me to repeat, emphatically, that Marley was as dead as a door-nail. " +
            "Scrooge knew he was dead? Of course he did. How could it be otherwise? Scrooge and he were partners for I don’t know how many years. " +
            "Scrooge was his sole executor, his sole administrator, his sole assign, his sole residuary legatee, his sole friend, and sole mourner. " +
            "And even Scrooge was not so dreadfully cut up by the sad event, but that he was an excellent man of business on the very day of the funeral, " +
            "and solemnised it with an undoubted bargain.";

    private StanfordCoreNLP pipeline;
    private CoreDocument document;

    public StanfordNlpAnalyzer(String text) {
        Properties props = new Properties();
        // set the list of annotators to run
        props.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner,parse,depparse,coref,kbp");
        // set a property for an annotator, in this case the coref annotator is being set to use the neural algorithm
        props.setProperty("coref.algorithm", "neural");
        // build pipeline
        pipeline = new StanfordCoreNLP(props);
        document = new CoreDocument(text);
        pipeline.annotate(document);
    }

    /**
     * NER - Named Entity Recognition.
     * For English, by default, the annotator (ner) recognizes named (PERSON, LOCATION, ORGANIZATION, MISC),
     * numerical (MONEY, NUMBER, ORDINAL, PERCENT), and temporal (DATE, TIME, DURATION, SET) entities (12 classes).
     * If entity was not recognized then it is marked with "O"
     * @return list of NER excluding "O"
     */
    public List<String> getNerList() {
        LOG.info("==== Started NER (Named Entity Recognition) ====");
        List<String> nerTags = new ArrayList<>();
        nerTags.addAll(document
                .tokens()
                .stream()
                .filter(token -> !token.ner().equals("O"))
                .map(token -> {
                    LOG.debug("Entity: " + token.originalText() + " Ner: " + token.ner() + " Tag: " + token.tag());
                    return token.ner();
                })
                .collect(Collectors.toList()));
        LOG.info("==== Finished NER ====");
        return nerTags;
    }

    /**
     * Weight of NER is how many times NER tag happens in text.
     * @return map of NERs with weights, where NER is a key and weight of NER is a value
     */
    public Map<String, Integer> getNerWeights() {
        return getNerList()
                .stream()
                .collect(Collectors.toMap(key -> key, value -> 1, (oldVal, newVal) -> oldVal + 1));
    }

    /**
     * KBP is Knowledge Base Population.
     * For example when run on the input sentence:
     *  Joe Smith was born in Oregon.
     * The annotator will find the following ("subject", "relation", "object") triple:
     *  ("Joe Smith", "per:stateorprovince_of_birth", "Oregon" }
     * @return list of {@RelationTriple}
     */
    public List<RelationTriple> getKbpList() {
        LOG.info("==== Started KBP  (relation triples) ====");
        List<RelationTriple> relationTripleList = new ArrayList<>();
        document
                .sentences()
                .forEach(
                        s -> s.relations().forEach(r -> {
                            LOG.debug("KBP for sentence: " + s.text());
                            LOG.debug("KBP:" + r.toString());
                            relationTripleList.add(r);
                        })
                );
        LOG.info("==== Finished KBP ====");
        return relationTripleList;
    }

    /**
     * The CorefAnnotator finds mentions of the same entity in a text, such as when “Theresa May” and “she” refer to the same person.
     */
    public void getCorefList(){
        LOG.info("==== Started Coref ====");
        document
                .corefChains()
                .forEach((k, v) -> LOG.info(k + ": " + v));
        LOG.info("==== Finished Coref ====");
    }


    public static void main(String[] args) {
        StanfordNlpAnalyzer a = new StanfordNlpAnalyzer(StanfordNlpAnalyzer.text);
        //getting ner weights
        LOG.info("NER weights:\n" + a.getNerWeights().toString() + "\n");
        //detecting the main ner (with the biggest weight)
        a.getNerWeights()
                .entrySet()
                .stream()
                .max(Comparator.comparingInt(Map.Entry::getValue))
                .ifPresent(e -> LOG.info("\nThe main NER tag for the text is " + e.getKey() + " with weight " + e.getValue() +"\n"));
        //getting kbp
        a.getKbpList().forEach(r -> LOG.info("\n" + r.toString() + "\n"));
        //getting coref
        a.getCorefList();
    }
}
