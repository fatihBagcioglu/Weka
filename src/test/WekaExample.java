package test;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.SMO;
import weka.core.Instance;

public class WekaExample {
    public static void main(String[] args) {
        try {
            // Veri kaynağını yükleme
            DataSource source = new DataSource("iris.arff");

            // Veri kümesini yükleme
            Instances data = source.getDataSet();
            if (data.classIndex() == -1) {
                data.setClassIndex(data.numAttributes() - 1);
            }

            // Sınıflandırma modelini oluşturma
            SMO classifier = new SMO();
            classifier.buildClassifier(data);

            // Yeni bir Instances nesnesi oluşturma
            Instances instances = new Instances(data);

            // Instance nesnesini almak için indeks belirleme (0, 1, 2, ...)
            int instanceIndex = 0;

            // Instance nesnesini al
            Instance instance = instances.instance(instanceIndex);

            // Örnek öznitelik değerlerini ayarlama
            instance.setValue(0, 5.1);
            instance.setValue(1, 3.5);
            instance.setValue(2, 1.4);
            instance.setValue(3, 0.2);
            

            // Tahmini yapma
            double prediction = classifier.classifyInstance(instance);
            String className = data.classAttribute().value((int) prediction);

            System.out.println("Tahmin edilen sınıf: " + className);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
}