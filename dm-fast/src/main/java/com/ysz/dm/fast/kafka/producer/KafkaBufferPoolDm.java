package com.ysz.dm.fast.kafka.producer;

import java.nio.ByteBuffer;
import org.apache.kafka.clients.producer.internals.BufferPool;
import org.apache.kafka.common.metrics.Metrics;
import org.apache.kafka.common.utils.Time;
import org.junit.Before;
import org.junit.Test;

public class KafkaBufferPoolDm {

  private BufferPool bufferPool = null;

  @Before
  public void setUp() {
    /*1024Kb = 1M 的总内存空间 */
    final long memory = 1L * 1024L * 1024L;
    /*1kb 的poolsize*/
    final int poolableSize = 1024;
    final Metrics metrics = new Metrics();
    Time time = Time.SYSTEM;
    String metricGrpName = "tstGrpName";

    this.bufferPool = new BufferPool(
        memory, poolableSize, metrics, time, metricGrpName
    );
  }


  @Test
  public void tstNormalAllocate() throws InterruptedException {
    /*刚好要 1个 */
    int size = 1025;
    long maxTimeToBlockMs = 10L;
    ByteBuffer byteBuffer = this.bufferPool.allocate(size, maxTimeToBlockMs);
    System.out.println("分配成功: " + byteBuffer);
  }


}
