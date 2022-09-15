package com.ysz.dm.ddd.vshop.domain.core.common.media;

import lombok.Getter;
import lombok.ToString;

/**
 * @author carl
 * @create 2022-09-15 11:39 AM
 **/
@ToString
@Getter
public class Media {

  private boolean main;

  private MediaType mediaType;

  private String bucketName;

  private String key;

  /**
   * 如果是 图片, 图片详情
   */
  private PicDetail picDetail;

  /**
   * 如果 type 是 video, 视频详情
   */
  private VideoDetail videoDetail;

  public String path() {
    return String.format("%s/%s", bucketName, key);
  }
}
