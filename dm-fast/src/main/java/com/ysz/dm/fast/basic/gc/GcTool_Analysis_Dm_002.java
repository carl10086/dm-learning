package com.ysz.dm.fast.basic.gc;

import com.microsoft.gctoolkit.GCToolKit;
import com.microsoft.gctoolkit.io.GCLogFile;
import com.microsoft.gctoolkit.io.SingleGCLogFile;
import com.microsoft.gctoolkit.jvm.JavaVirtualMachine;
import com.ysz.dm.fast.basic.gc.aggregation.PauseTimeSummary;
import java.nio.file.Path;
import java.nio.file.Paths;

public class GcTool_Analysis_Dm_002 {


  public void execute() throws Exception {
    final Path path = Paths.get("/Users/carl/tmp/fuck/gc-saturn.log.0.current");
//    final Path path = Paths.get("/Users/carl/tmp/fuck/gc.log");
    System.out.println(path);
    GCLogFile gcLogFile = new SingleGCLogFile(path);

    final GCToolKit gcToolKit = new GCToolKit();
    gcToolKit.loadAggregationsFromServiceLoader();

    final JavaVirtualMachine machine = gcToolKit.analyze(gcLogFile);

    machine.getAggregation(PauseTimeSummary.class).ifPresent(pauseTimeSummary -> {
      System.out.printf("Total pause time  : %.4f\n", pauseTimeSummary.getTotalPauseTime());
      System.out.printf("Total run time    : %.4f\n", pauseTimeSummary.getRuntimeDuration());
      System.out.printf("Percent pause time: %.2f\n", pauseTimeSummary.getPercentPaused());
    });


  }


  public static void main(String[] args) throws Exception {
    new GcTool_Analysis_Dm_002().execute();
  }

}
