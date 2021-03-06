package com.konakartadmin.app;

import com.konakartadmin.appif.*;
import com.konakartadmin.bl.KKAdmin;

/**
 *  The KonaKart Custom Engine - GetContent - Generated by CreateKKAdminCustomEng
 */
@SuppressWarnings("all")
public class GetContent
{
    KKAdmin kkAdminEng = null;

    /**
     * Constructor
     */
     public GetContent(KKAdmin _kkAdminEng)
     {
         kkAdminEng = _kkAdminEng;
     }

     public AdminContent getContent(String sessionId, int contentId) throws KKAdminException
     {
         return kkAdminEng.getContent(sessionId, contentId);
     }
}
