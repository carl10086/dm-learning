package com.ysz.dm.netty.dm.lettuce;

import brave.Tracing;
import com.google.common.collect.Lists;
import io.lettuce.core.ReadFrom;
import io.lettuce.core.RedisClient;
import io.lettuce.core.RedisURI;
import io.lettuce.core.api.sync.RedisCommands;
import io.lettuce.core.codec.StringCodec;
import io.lettuce.core.masterreplica.MasterReplica;
import io.lettuce.core.masterreplica.StatefulRedisMasterReplicaConnection;
import io.lettuce.core.metrics.MicrometerCommandLatencyRecorder;
import io.lettuce.core.metrics.MicrometerOptions;
import io.lettuce.core.resource.ClientResources;
import io.lettuce.core.tracing.BraveTracing;
import io.micrometer.core.instrument.logging.LoggingMeterRegistry;
import java.util.Map;
import lombok.extern.slf4j.Slf4j;
import zipkin2.Endpoint;

@Slf4j
public class BasicUsageDm {

  public static void main(String[] args) throws Exception {

    brave.Tracing clientTracing = Tracing.newBuilder()
        .localServiceName("my-redis")
        .spanReporter(span -> {
          final Map<String, String> tags = span.tags();
          final Endpoint endpoint = span.remoteEndpoint();
          log.info("endpoint:{}, cmd:{}, args:{}", endpoint, tags.getOrDefault("cmd", "noCmd"),
              tags.getOrDefault("args", "noArgs"));
        })
        .build();

    BraveTracing tracing = BraveTracing.builder().tracing(clientTracing)
        .excludeCommandArgsFromSpanTags()
        .serviceName("custom-service-name-goes-here")
        .spanCustomizer((command, span) -> span.tag("cmd", command.getType().name())
            .tag("args", command.getArgs() == null ? "" : command.getArgs().toCommandString()))
        .build();

    LoggingMeterRegistry meterRegistry = new LoggingMeterRegistry();
    MicrometerOptions options = MicrometerOptions.create();
    ClientResources resources = ClientResources.builder().commandLatencyRecorder(new MicrometerCommandLatencyRecorder(
        meterRegistry, options
    )).tracing(tracing).build();
    RedisClient redisClient = RedisClient.create(resources);

//    final EventBus eventBus = redisClient.getResources().eventBus();

//    eventBus.get().filter(event -> event instanceof ConnectionActivatedEvent)
//        .cast(ConnectionActivatedEvent.class).subscribe(e -> System.out.println(e));
//
    StatefulRedisMasterReplicaConnection<String, String> connection = MasterReplica.connect(
        redisClient, StringCodec.UTF8,
        Lists.newArrayList(
            RedisURI.Builder.redis("10.200.64.4", 6379).withDatabase(0).build(),
            RedisURI.Builder.redis("10.200.64.4", 6379).withDatabase(0).build()
        )
    );

    connection.setReadFrom(ReadFrom.REPLICA);
//    connection.setReadFrom(ReadFrom.UPSTREAM);

    final RedisCommands<String, String> sync = connection.sync();
    System.err.println(sync.get("username"));
    sync.set("username", "bbbb");
    System.err.println(sync.get("username"));

    connection.close();

    redisClient.shutdown();
//    System.in.read();

  }

}
