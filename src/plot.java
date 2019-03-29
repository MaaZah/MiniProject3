import javax.swing.JFrame;
import javax.swing.SwingUtilities;
import javax.swing.WindowConstants;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;


public class plot extends JFrame {
	  private static final long serialVersionUID = 6294689542092367723L;

	  public plot(String title, double[][] mat) {
	    super(title);

	    // Create dataset
	    XYDataset dataset = createDataset(mat);

	    // Create chart
	    JFreeChart chart = ChartFactory.createXYLineChart(
	        "SSE vs Iteration",
	        "Iteration",
	        "SSE",
	        dataset,
	        PlotOrientation.VERTICAL,
	        true, true, false);

	    // Create Panel
	    ChartPanel panel = new ChartPanel(chart);
	    setContentPane(panel);
	  }

	  private XYDataset createDataset(double[][] mat) {
	    XYSeriesCollection dataset = new XYSeriesCollection();


	    XYSeries series = new XYSeries("trendline");
	    
	    for(int i = 0; i< mat.length;i++){
	    	series.add(mat[i][0], mat[i][1]);
	    }

	    //Add series to dataset
	    dataset.addSeries(series);
	    
	    return dataset;
	  }
	}