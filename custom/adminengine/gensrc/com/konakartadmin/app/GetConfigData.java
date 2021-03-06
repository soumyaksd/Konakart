package com.konakartadmin.app;

import com.konakartadmin.appif.*;
import com.konakartadmin.bl.KKAdmin;

/**
 *  The KonaKart Custom Engine - GetConfigData - Generated by CreateKKAdminCustomEng
 */
@SuppressWarnings("all")
public class GetConfigData
{
    KKAdmin kkAdminEng = null;

    /**
     * Constructor
     */
     public GetConfigData(KKAdmin _kkAdminEng)
     {
         kkAdminEng = _kkAdminEng;
     }

     public AdminConfigData[] getConfigData(String sessionId, String key) throws KKAdminException
     {
         return kkAdminEng.getConfigData(sessionId, key);
     }
}
