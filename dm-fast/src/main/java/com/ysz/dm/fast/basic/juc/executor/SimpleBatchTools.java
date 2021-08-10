package com.duitang.webscript.infra.utils;

import com.google.common.base.Preconditions;
import java.io.File;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.LineIterator;

@Slf4j
public class SimpleBatchTools {


  public static void batchWithLargeFile(
      final File file,
      final int batchSize,
      final Consumer<List<String>> consumer
  ) {
    batchWithLargeFile(file, batchSize, consumer, false);
  }

  /**
   * 遍历一个文件、按行进行批处理
   *
   * @param file           文件名称
   * @param batchSize      批处理大小
   * @param consumer       调用方逻辑
   * @param onlyFirstBatch 如果为 true、会只对第一批处理、用于测试
   */
  public static void batchWithLargeFile(
      final File file,
      final int batchSize,
      final Consumer<List<String>> consumer,
      final boolean onlyFirstBatch
  ) {
    Preconditions.checkArgument(batchSize > 0);
    LineIterator it = null;
    try {
      it = FileUtils.lineIterator(file, Charset.defaultCharset().displayName());
      int count = 0;
      final List<String> buffer = new ArrayList<>(batchSize);
      while (it.hasNext()) {
        count++;
        final String line = it.next();
        buffer.add(line);
        if (count % batchSize == 0) {
          consumer.accept(buffer);
          buffer.clear();
          /*测试的时候可能只想要测试第一批*/
          if (onlyFirstBatch) {
            break;
          }
        }
      }
      if (buffer.size() > 0) {
        consumer.accept(buffer);
        buffer.clear();
      }

    } catch (Exception e) {
      log.error("iterator file failed:{}, batchSize:{}", file, batchSize, e);
    } finally {
      if (it != null) {
        LineIterator.closeQuietly(it);
      }
    }
  }

  public static <T> void batchWithCollection(
      final Iterable<T> collection,
      final int batchSize,
      final Consumer<List<T>> consumer
  ) {
    batchWithCollection(collection, batchSize, consumer, false);
  }


  public static <T> void batchWithCollection(
      final Iterable<T> collection,
      final int batchSize,
      final Consumer<List<T>> consumer,
      final boolean onlyFirstBatch
  ) {
    if (collection == null) {
      return;
    }
    Preconditions.checkArgument(batchSize > 0);
    final List<T> buffer = new ArrayList<>(batchSize);
    int count = 0;
    for (T t : collection) {
      buffer.add(t);
      count++;
      if (count % batchSize == 0) {
        consumer.accept(buffer);
        buffer.clear();
        if (onlyFirstBatch) {
          break;
        }
      }
    }
    if (buffer.size() > 0) {
      consumer.accept(buffer);
      buffer.clear();
    }
  }
}
