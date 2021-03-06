package com.konakartadmin.app;

import com.konakartadmin.bl.KKAdmin;

/**
 *  The KonaKart Custom Engine - RemoveConfigs - Generated by CreateKKAdminCustomEng
 */
@SuppressWarnings("all")
public class RemoveConfigs
{
    KKAdmin kkAdminEng = null;

    /**
     * Constructor
     */
     public RemoveConfigs(KKAdmin _kkAdminEng)
     {
         kkAdminEng = _kkAdminEng;
     }

     public void removeConfigs(String sessionId, String[] configKeys) throws KKAdminException
     {
         kkAdminEng.removeConfigs(sessionId, configKeys);
     }
}
