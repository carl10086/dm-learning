package com.ysz.dm.duitang.srv.support.blog.port.adapter.persist.impl;

import com.ysz.dm.duitang.srv.support.blog.domain.model.Blog;
import com.ysz.dm.duitang.srv.support.blog.domain.repo.BlogRepo;
import com.ysz.dm.duitang.srv.support.blog.port.adapter.persist.converter.BlogConverter;
import com.ysz.dm.duitang.srv.support.blog.port.adapter.persist.dao.BlogForwardDAO;
import com.ysz.dm.duitang.srv.support.blog.port.adapter.persist.dao.BlogMetaDAO;
import com.ysz.dm.duitang.srv.support.blog.port.adapter.persist.dataobject.BlogForwardDO;
import com.ysz.dm.duitang.srv.support.blog.port.adapter.persist.dataobject.BlogMetaDO;
import com.ysz.dm.duitang.srv.support.blog.port.adapter.persist.mapper.BlogMapper;

public class BlogRepoImpl implements BlogRepo {

  private BlogForwardDAO blogForwardDAO;

  private BlogMetaDAO blogMetaDAO;

  private BlogConverter blogConverter;

  private BlogMapper blogMapper;

  @Override
  public void save(final Blog blog) {
    BlogForwardDO blogForwardDO = blogMapper.toBlogForwardDO(blog);
    BlogMetaDO blogMetaDO = blogMapper.toBlogMetaDO(blog.getBlogMeta());

    /*如果是 Mysql 同一个事务即可*/

    blogMetaDAO.insertOne(blogMetaDO); /*如果没有事务、先插入 meta，根据 meta 定时修复 对应的 forward 追求最终一致性*/
    blogForwardDAO.insertOne(blogForwardDO);
  }

  @Override
  public Blog findOne(final Long forwardId) {
    BlogForwardDO blogForwardDO = blogForwardDAO.selectOneById(forwardId);
    BlogMetaDO blogMetaDO = blogMetaDAO.selectOneById(blogForwardDO.getRootId());
    return blogConverter.toBlogForward(blogForwardDO, blogMetaDO);
  }

  @Override
  public void addForward(final Blog blog) {

  }

  @Override
  public void changeAuditStatus(final Blog blog) {

  }
}
