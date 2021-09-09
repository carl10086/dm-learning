package com.ysz.dm.netty.custom.sofa.bolt.remoting.server;

import com.ysz.dm.netty.custom.sofa.bolt.remoting.cfg.MyBoltGenericOptions;
import com.ysz.dm.netty.custom.sofa.bolt.remoting.cfg.MyBoltServerOptions;
import com.ysz.dm.netty.custom.sofa.bolt.remoting.core.address.MyAddressParser;
import com.ysz.dm.netty.custom.sofa.bolt.remoting.core.address.MyAddressParserImpl;
import com.ysz.dm.netty.custom.sofa.bolt.remoting.core.connection.MyConnectionManager;
import com.ysz.dm.netty.custom.sofa.bolt.remoting.core.connection.impl.MyDefaultConnectionManager;
import com.ysz.dm.netty.custom.sofa.bolt.remoting.core.connection.impl.MyRandomSelectStrategy;
import com.ysz.dm.netty.custom.sofa.bolt.remoting.infra.thread.MyNamedThreadFactory;
import com.ysz.dm.netty.custom.sofa.bolt.remoting.infra.utils.MyNettyEventLoopUtils;
import io.netty.bootstrap.ServerBootstrap;
import io.netty.buffer.PooledByteBufAllocator;
import io.netty.buffer.UnpooledByteBufAllocator;
import io.netty.channel.Channel;
import io.netty.channel.ChannelFuture;
import io.netty.channel.ChannelInitializer;
import io.netty.channel.ChannelOption;
import io.netty.channel.ChannelPipeline;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.WriteBufferWaterMark;

public class MyRpcServer extends MyAbstractRemotingServer {

  private ServerBootstrap bootstrap;
  private ChannelFuture channelFuture;
  private MyAddressParser myAddressParser;
  private MyConnectionManager connectionManager;


  private final EventLoopGroup boss = MyNettyEventLoopUtils.newEventLoopGroup(
      1,
      new MyNamedThreadFactory("Rpc-netty-server-boss", false)
  );

  private final EventLoopGroup worker = MyNettyEventLoopUtils.newEventLoopGroup(
      Runtime.getRuntime().availableProcessors() * 2,
      new MyNamedThreadFactory("Rpc-netty-server-worker", true)
  );


  public MyRpcServer(
      String ip,
      int port) {
    this(ip, port, false, false);
  }

  @Override
  protected void doInit() {
    initThenFrozenCfg();
    initAddressParser();

    if (getOption(MyBoltServerOptions.SERVER_MANAGE_CONNECTION_SWITCH)) {
      final MyRandomSelectStrategy connectionSelectStrategy = new MyRandomSelectStrategy(
          super.options);
      this.connectionManager = new MyDefaultConnectionManager();
    } else {

    }

    this.bootstrap = new ServerBootstrap();
    initBootstrap();
    initWaterMark();
    initAllocator();

    final boolean idleSwitch = getOption(MyBoltGenericOptions.TCP_IDLE_SWITCH);
    final boolean flushConsolidationSwitch = getOption(
        MyBoltGenericOptions.NETTY_FLUSH_CONSOLIDATION);

    final int idleTime = getOption(MyBoltServerOptions.TCP_SERVER_IDLE);
    this.bootstrap.childHandler(new ChannelInitializer<>() {
      @Override
      protected void initChannel(final Channel channel) throws Exception {
        final ChannelPipeline pipeline = channel.pipeline();
        if (flushConsolidationSwitch) {

        }
      }
    });

  }

  private void initAllocator() {
    if (getOption(MyBoltGenericOptions.NETTY_BUFFER_POOLED)) {
      this.bootstrap.option(ChannelOption.ALLOCATOR, PooledByteBufAllocator.DEFAULT)
          .childOption(ChannelOption.ALLOCATOR, PooledByteBufAllocator.DEFAULT);
    } else {
      this.bootstrap.option(ChannelOption.ALLOCATOR, UnpooledByteBufAllocator.DEFAULT)
          .childOption(ChannelOption.ALLOCATOR, UnpooledByteBufAllocator.DEFAULT);
    }
  }

  private void initWaterMark() {
    this.bootstrap.childOption(
        ChannelOption.WRITE_BUFFER_WATER_MARK, new WriteBufferWaterMark(
            getOption(MyBoltGenericOptions.NETTY_BUFFER_LOW_WATER_MARK),
            getOption(MyBoltGenericOptions.NETTY_BUFFER_HIGH_WATER_MARK)
        )
    );
  }

  private ServerBootstrap initBootstrap() {
    return this.bootstrap.group(boss, worker)
        .channel(MyNettyEventLoopUtils.getServerSocketChannelClass())
        .option(ChannelOption.SO_BACKLOG, getOption(MyBoltServerOptions.TCP_SO_BACKLOG))
        .option(ChannelOption.SO_REUSEADDR, getOption(MyBoltGenericOptions.TCP_SO_REUSEADDR))
        .option(ChannelOption.TCP_NODELAY, getOption(MyBoltGenericOptions.TCP_NODELAY))
        .option(ChannelOption.SO_KEEPALIVE, getOption(MyBoltGenericOptions.TCP_SO_KEEPALIVE));
  }

  private void initThenFrozenCfg() {

  }

  private void initAddressParser() {
    if (this.myAddressParser == null) {
      this.myAddressParser = new MyAddressParserImpl();
    }
  }

  public MyRpcServer(
      String ip,
      int port,
      boolean manageConnection,
      boolean syncStop
  ) {
    super(ip, port);
    super
        .updateOption(MyBoltServerOptions.SERVER_MANAGE_CONNECTION_SWITCH, manageConnection)
        .updateOption(MyBoltServerOptions.SERVER_SYNC_STOP, syncStop);
  }

}
