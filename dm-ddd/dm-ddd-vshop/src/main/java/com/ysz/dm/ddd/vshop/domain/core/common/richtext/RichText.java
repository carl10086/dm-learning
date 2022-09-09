package com.ysz.dm.ddd.vshop.domain.core.common.richtext;

import lombok.Getter;
import lombok.ToString;

/**
 * 富文本类型 .
 *
 * content 和  type 决定了 文本的渲染方式
 *
 * @author carl
 * @create 2022-09-09 10:30 AM
 **/
@ToString
@Getter
public class RichText {

  private RichTextType type;

  private String content;
  private Long createAt;

  private Long updateAt;


  private String updateBy;

  public RichText setType(RichTextType type) {
    this.type = type;
    return this;
  }

  public RichText setCreateAt(Long createAt) {
    this.createAt = createAt;
    return this;
  }

  public RichText setUpdateAt(Long updateAt) {
    this.updateAt = updateAt;
    return this;
  }

  public RichText setContent(String content) {
    this.content = content;
    return this;
  }

  public RichText setUpdateBy(String updateBy) {
    this.updateBy = updateBy;
    return this;
  }
}
