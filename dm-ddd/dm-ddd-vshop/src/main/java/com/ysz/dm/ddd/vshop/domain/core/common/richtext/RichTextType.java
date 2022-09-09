package com.ysz.dm.ddd.vshop.domain.core.common.richtext;

/**
 * 富文本类型, 决定了 渲染类型
 *
 * @author carl
 * @create 2022-09-09 10:30 AM
 **/
public enum RichTextType {
  /**
   * markdown 格式的富文本
   */
  markdown,

  /**
   * 是什么就显示什么
   */
  simple,
  /**
   * html 形式的富文本
   */
  html
}
