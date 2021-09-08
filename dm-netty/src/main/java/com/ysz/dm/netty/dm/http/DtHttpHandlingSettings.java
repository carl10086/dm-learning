package com.ysz.dm.netty.dm.http;

import lombok.Getter;
import lombok.ToString;

@ToString
@Getter
public class DtHttpHandlingSettings {


  private final int maxContentLength;
  private final int maxChunkSize;
  private final int maxHeaderSize;
  private final int maxInitialLineLength;
  private final boolean resetCookies;
  private final boolean compression;
  private final int compressionLevel;
  private final boolean detailedErrorsEnabled;
  private final int pipeliningMaxEvents;
  private final long readTimeoutMillis;
  private boolean corsEnabled;

  public DtHttpHandlingSettings(
      int maxContentLength,
      int maxChunkSize,
      int maxHeaderSize,
      int maxInitialLineLength,
      boolean resetCookies,
      boolean compression,
      int compressionLevel,
      boolean detailedErrorsEnabled,
      int pipeliningMaxEvents,
      long readTimeoutMillis,
      boolean corsEnabled) {
    this.maxContentLength = maxContentLength;
    this.maxChunkSize = maxChunkSize;
    this.maxHeaderSize = maxHeaderSize;
    this.maxInitialLineLength = maxInitialLineLength;
    this.resetCookies = resetCookies;
    this.compression = compression;
    this.compressionLevel = compressionLevel;
    this.detailedErrorsEnabled = detailedErrorsEnabled;
    this.pipeliningMaxEvents = pipeliningMaxEvents;
    this.readTimeoutMillis = readTimeoutMillis;
    this.corsEnabled = corsEnabled;
  }


}
