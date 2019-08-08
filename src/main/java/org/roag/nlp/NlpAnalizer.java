package org.roag.nlp;

import edu.stanford.nlp.ie.util.RelationTriple;
import edu.stanford.nlp.pipeline.*;

import java.util.*;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;

/**
 * Created by eurohlam on 7/08/2019.
 * Some more deep experiments with Stanford CoreNLP API
 */
public class NlpAnalizer {

    private static final Logger LOG = Logger.getLogger(NlpAnalizer.class.getName());

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

    public NlpAnalizer(String text) {
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

    public List<String> getNerList() {
        LOG.info("==== Started NER (Named Entity Recognition) ====");
        List<String> nerTags = new ArrayList<>();
        nerTags.addAll(document
                .tokens()
                .stream()
                .filter(token -> !token.ner().equals("O"))
                .map(token -> {
                    LOG.fine("Word: " + token.originalText() + " Ner: " + token.ner() + " Tag: " + token.tag());
                    return token.ner();
                })
                .collect(Collectors.toList()));
        LOG.info("==== Finished NER ====");
        return nerTags;
    }

    /**
     * Weight of ner is how many times ner tag happens in text.
     * @return map of ners with weights, whee key is ner and value is weight
     */
    public Map<String, Integer> getNerWeights() {
        return getNerList()
                .stream()
                .collect(Collectors.toMap(key -> key, value -> 1, (oldVal, newVal) -> oldVal + 1));
    }

    public List<RelationTriple> getKbpList() {
        LOG.info("==== Started KBP  (relation triples) ====");
        List<RelationTriple> relationTripleList = new ArrayList<>();
        document
                .sentences()
                .forEach(
                        s -> s.relations().forEach(r -> {
                            LOG.fine("KBP for sentence: " + s.text());
                            LOG.fine("KBP:" + r.toString());
                            relationTripleList.add(r);
                        })
                );
        LOG.info("==== Finished KBP ====");
        return relationTripleList;
    }

    public void getCorefList(){
        LOG.info("==== Started Coref ====");
        document
                .corefChains()
                .forEach((k, v) -> LOG.info(k + ": " + v));
        LOG.info("==== Finished Coref ====");
    }


    public static void main(String[] args) {
        NlpAnalizer a = new NlpAnalizer(NlpAnalizer.text);
        //getting ner weights
        LOG.info("\n" + a.getNerWeights().toString() + "\n");
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
