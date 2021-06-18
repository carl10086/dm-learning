package com.ysz.dm.netty.http;

import io.netty.channel.Channel;
import io.netty.channel.ChannelInitializer;
import io.netty.handler.codec.ByteToMessageDecoder;
import io.netty.handler.codec.http.HttpContentDecompressor;
import io.netty.handler.codec.http.HttpRequestDecoder;
import io.netty.handler.codec.http.HttpResponseEncoder;

public class DtHttpChannelInitializer extends ChannelInitializer<Channel> {

  private int maxInitialLineLength = 8 * 1024;

  private int maxHeaderSize = 8 * 1024;

  private int maxChunkSize = 8 * 1024;

  private DtHttpHandlingSettings handlingSettings;

//  private DtHttpServerRestHandler restHandler;

  @Override
  protected void initChannel(final Channel ch) throws Exception {
    /*1. 创建一个 decoder*/
    HttpRequestDecoder decoder = new HttpRequestDecoder();
    /*Es 显示指定 cumulator */
    decoder.setCumulator(ByteToMessageDecoder.COMPOSITE_CUMULATOR);
    ch.pipeline().addLast("decoder", decoder);
    ch.pipeline().addLast("decoder_compress", new HttpContentDecompressor());
    ch.pipeline().addLast("encoder", new HttpResponseEncoder());
//    final HttpObjectAggregator aggregator = new HttpObjectAggregator(
//        handlingSettings.getMaxContentLength());
//    ch.pipeline().addLast("aggregator", aggregator);
//    if (handlingSettings.isCompression()) {
//      ch.pipeline().addLast("encoder_compress",
//          new HttpContentCompressor(handlingSettings.getCompressionLevel()));
//    }
    // cors 支持
    // pipeline 支持
//    ch.pipeline().addLast("pipelining", new Netty4HttpPipeliningHandler(logger, transport.pipeliningMaxEvents));

    ch.pipeline().addLast("handler", new HttpSnoopServerHandler());
  }
}
