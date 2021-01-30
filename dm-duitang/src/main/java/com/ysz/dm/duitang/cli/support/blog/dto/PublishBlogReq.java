package com.ysz.dm.duitang.cli.support.blog.dto;

import java.io.Serializable;
import lombok.Data;

@Data
public class PublishBlogReq implements Serializable {

  /**
   * 原发文本
   */
  private String msg;

  /**
   * 类型
   */
  private String contentType;

  /**
   * 作者
   */
  private Long userId;

  /**
   * 内容 id
   */
  private Long photoId;

}
