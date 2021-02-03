package com.ysz.dm.duitang.cli.support.blog.dto;

import java.io.Serializable;
import lombok.Data;

@Data
public class AuditBlogReq implements Serializable {


  private Long operateId;
  private Long blogId;


  /**
   * curStatus 和 curMsg 都是为了保证 运营眼中的数据和数据库中的一致
   *
   *
   * 可以用乐观锁替代、解决 ABA 问题 ? 这里跟 ddd 无关
   *
   */

  private Integer curStatus;
  private String curMsg;

}
