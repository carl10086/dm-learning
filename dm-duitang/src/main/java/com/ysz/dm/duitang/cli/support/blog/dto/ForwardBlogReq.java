package com.ysz.dm.duitang.cli.support.blog.dto;

import java.io.Serializable;
import lombok.Data;

@Data
public class ForwardBlogReq implements Serializable {


  /**
   * 作者
   */
  private Long userId;


  /**
   * 要 forward 的 blogId
   */
  private Long parentId;


}
