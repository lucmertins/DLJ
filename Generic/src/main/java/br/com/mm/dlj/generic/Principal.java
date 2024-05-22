package br.com.mm.dlj.generic;

import org.knowm.xchart.BitmapEncoder;
import org.knowm.xchart.QuickChart;
import org.knowm.xchart.XYChart;

/**
 *
 * @author mertins
 */
public class Principal {

    private void grafico() {
        double[] xData = new double[]{0.0, 1.0, 2.0};
        double[] yData = new double[]{2.0, 1.0, 0.0};

        XYChart chart = QuickChart.getChart("Sample Chart", "X", "Y", "y(x)", xData, yData);
        BitmapEncoder.getBufferedImage(chart);
    }

    public static void main(String[] args) {
        new Principal().grafico();
    }
}
