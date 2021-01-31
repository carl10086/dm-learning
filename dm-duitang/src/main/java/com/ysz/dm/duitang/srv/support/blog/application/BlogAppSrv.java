package com.ysz.dm.duitang.srv.support.blog.application;


import com.ysz.dm.duitang.cli.support.blog.dto.ForwardBlogReq;
import com.ysz.dm.duitang.cli.support.blog.dto.PublishBlogReq;
import com.ysz.dm.duitang.srv.infra.core.AccurateTimeManager;
import com.ysz.dm.duitang.srv.infra.core.IdGenerateManager;
import com.ysz.dm.duitang.srv.support.blog.domain.BlogContentType;
import com.ysz.dm.duitang.srv.support.blog.domain.BlogForward;
import com.ysz.dm.duitang.srv.support.blog.domain.BlogForwardNotForbiddenException;
import com.ysz.dm.duitang.srv.support.blog.domain.BlogForwardRepo;
import com.ysz.dm.duitang.srv.support.blog.domain.BlogMeta;
import com.ysz.dm.duitang.srv.support.blog.domain.BlogMetaRepo;
import com.ysz.dm.duitang.srv.support.blog.domain.BlogText;
import com.ysz.dm.duitang.srv.support.blog.domain.BlogUser;
import java.util.Date;

//@Slf4j
public class BlogAppSrv {

  private BlogMetaRepo blogMetaRepo;

  private BlogForwardRepo blogForwardRepo;

  private AccurateTimeManager accurateTimeManager;

  private IdGenerateManager idGenerateManager;

  private BlogMeta newBlogMeta(PublishBlogReq req) {
    final Date now = accurateTimeManager.now();
    return new BlogMeta(
        idGenerateManager.nextBlogId(),
        new BlogText(req.getMsg()),
        null /*extra 假装已经有了*/,
        now,
        BlogContentType.valueOf(req.getContentType()),
        new BlogUser(req.getUserId()),
        req.getPhotoId()
    );
  }

  public BlogForward publishBlog(PublishBlogReq req) {
    final BlogMeta blogMeta = newBlogMeta(req);
    blogMetaRepo.save(blogMeta);

    /**
     * 在生成一个原发(BlogMeta)的时候、也要生成一个默认的转发关系作为根(BlogForward)
     */
    BlogForward forward = new BlogForward(
        blogMeta.getId(),
        blogMeta,
        null,
        blogMeta.getBlogUser(),
        blogMeta.getCreateAt()
    );

    /**
     * 这里有 blog_meta 和 blog_forward 的一致性问题
     *
     * 可以追求最终一致性: 定时任务根据 blog_meta 去修复 blog_forward
     */
    blogForwardRepo.save(forward);
    return forward;
  }


  public BlogForward forwardBlog(ForwardBlogReq req) {
    BlogForward parent = findBlogForward(req.getParentId());

    if (!parent.canBeForward()) {
      throw new BlogForwardNotForbiddenException(req);
    }
    return doForwardBlog(parent, new BlogUser(req.getUserId()));
  }

  private BlogForward doForwardBlog(
      final BlogForward parent,
      final BlogUser blogUser) {
    final BlogMeta blogMeta = parent.getBlogMeta();
    final BlogForward blogForward = new BlogForward(
        idGenerateManager.nextBlogId(),
        blogMeta,
        parent.getId(),
        blogUser,
        accurateTimeManager.now()
    );
    blogForwardRepo.save(blogForward);
    return blogForward;
  }

  private BlogForward findBlogForward(final Long forwardId) {
   return blogForwardRepo.findOne(forwardId);
  }


}
