package com.ysz.biz;

import com.google.common.base.Splitter;
import java.util.List;

public class Tmp {

  private static final String SOURCE = "mapreduce.input.fileinputformat.split.maxsize=750000000\n"
      + "hive.vectorized.execution.enabled=true\n"
      + "\n"
      + "hive.cbo.enable=true\n"
      + "hive.optimize.reducededuplication.min.reducer=4\n"
      + "hive.optimize.reducededuplication=true\n"
      + "hive.orc.splits.include.file.footer=false\n"
      + "hive.merge.mapfiles=true\n"
      + "hive.merge.sparkfiles=false\n"
      + "hive.merge.smallfiles.avgsize=16000000\n"
      + "hive.merge.size.per.task=256000000\n"
      + "hive.merge.orcfile.stripe.level=true\n"
      + "hive.auto.convert.join=true\n"
      + "hive.auto.convert.join.noconditionaltask=true\n"
      + "hive.auto.convert.join.noconditionaltask.size=894435328\n"
      + "hive.optimize.bucketmapjoin.sortedmerge=false\n"
      + "hive.map.aggr.hash.percentmemory=0.5\n"
      + "hive.map.aggr=true\n"
      + "hive.optimize.sort.dynamic.partition=false\n"
      + "hive.stats.autogather=true\n"
      + "hive.stats.fetch.column.stats=true\n"
      + "hive.vectorized.execution.reduce.enabled=false\n"
      + "hive.vectorized.groupby.checkinterval=4096\n"
      + "hive.vectorized.groupby.flush.percent=0.1\n"
      + "hive.compute.query.using.stats=true\n"
      + "hive.limit.pushdown.memory.usage=0.4\n"
      + "hive.optimize.index.filter=true\n"
      + "hive.exec.reducers.bytes.per.reducer=67108864\n"
      + "hive.smbjoin.cache.rows=10000\n"
      + "hive.exec.orc.default.stripe.size=67108864\n"
      + "hive.fetch.task.conversion=more\n"
      + "hive.fetch.task.conversion.threshold=1073741824\n"
      + "hive.fetch.task.aggr=false\n"
      + "mapreduce.input.fileinputformat.list-status.num-threads=5\n"
      + "spark.kryo.referenceTracking=false\n"
      + "spark.kryo.classesToRegister=org.apache.hadoop.hive.ql.io.HiveKey,org.apache.hadoop.io.BytesWritable,org.apache.hadoop.hive.ql.exec.vector.VectorizedRowBatch";

  public static void main(String[] args) {
    final List<String> strings = Splitter.on("\n").trimResults().omitEmptyStrings().splitToList(SOURCE);
    for (String string : strings) {
      final String[] split = string.split("=");
      if (split.length == 2) {
        System.out.println("  <property>\n"
            + "    <name>" + split[0].trim() + "</name>\n"
            + "    <value>" + split[1].trim() + "</value>\n"
            + "  </property>\n");
      }
    }
  }
}
